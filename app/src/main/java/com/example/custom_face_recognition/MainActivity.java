package com.example.custom_face_recognition;

import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.media.Image;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.util.Rational;
import android.view.Surface;
import android.view.TextureView;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraX;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageAnalysisConfig;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.PreviewConfig;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.firebase.ml.vision.common.FirebaseVisionImage;
import com.google.firebase.ml.vision.common.FirebaseVisionImageMetadata;
import com.google.firebase.ml.vision.face.FirebaseVisionFace;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_face;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.List;

import static org.opencv.core.CvType.CV_32SC1;


public class MainActivity extends AppCompatActivity {

    private int REQUEST_CODE_PERMISSIONS = 101;
    private final String[] REQUIRED_PERMISSIONS = new String[]{"android.permission.CAMERA", "android.permission.WRITE_EXTERNAL_STORAGE"};
    TextureView textureView;
    ImageView ivBitmap;
    private FacePredictor facePredictor;

    ImageAnalysis imageAnalysis;
    Preview preview;
    TextView statusView;
    public static final String TAG = "LogCat";
    FaceTracker faceTracker;
    FaceRecognizer faceRecognizer;

    Scalar blackColor = new Scalar(0, 0, 0, 255);
    Scalar greenColor = new Scalar(0, 255, 0, 255);
    Scalar redColor = new Scalar(255, 0, 0, 255);
    Scalar yellowColor = new Scalar(255, 255, 0, 255);
    Scalar whiteColor = new Scalar(255, 255, 255, 255);
    final int FACE_FRAMES_2_TRAIN = 10, FACE_FRAMES_2_TEST = 5;


    //private opencv_face.FaceRecognizer mLBPHFaceRecognizer;
//    float lbphThresh;

    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    initializeOpenCVDependencies();
                    break;
                }
                default: {
                    super.onManagerConnected(status);
                    break;
                }
            }
        }
    };

    private void initializeOpenCVDependencies() {

        try {
            facePredictor = new FacePredictor();
            faceRecognizer = new FaceRecognizer();

            if (allPermissionsGranted()) {
                startCamera();
            } else {
                ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS);
            }

        } catch (Exception e) {
            Log.e("OpenCVActivity", "Error loading cascade", e);
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        textureView = findViewById(R.id.textureView);
        ivBitmap = findViewById(R.id.ivBitmap);
        statusView = findViewById(R.id.status);
    }

    private void startCamera() {

        CameraX.unbindAll();
        preview = setPreview();
        imageAnalysis = setImageAnalysis();

        CameraX.bindToLifecycle(this, preview, imageAnalysis);
    }

    private Preview setPreview() {

        Rational aspectRatio = new Rational(textureView.getWidth(), textureView.getHeight());
        android.util.Size screen = new android.util.Size(textureView.getWidth(), textureView.getHeight());

        PreviewConfig pConfig = new PreviewConfig.Builder().setTargetAspectRatio(aspectRatio).setTargetResolution(screen).build();
        Preview preview = new Preview(pConfig);

        preview.setOnPreviewOutputUpdateListener(
                output -> {
                    ViewGroup parent = (ViewGroup) textureView.getParent();
                    parent.removeView(textureView);
                    parent.addView(textureView, 0);
                    textureView.setSurfaceTexture(output.getSurfaceTexture());
                    updateTransform();
                });


        return preview;
    }

    ImageAnalysis.Analyzer analyzer = new ImageAnalysis.Analyzer() {
        @Override
        public void analyze(ImageProxy imageProxy, int rotationDegrees) {
            if (imageProxy == null || imageProxy.getImage() == null) {
                return;
            }
            Image mediaImage = imageProxy.getImage();

            int rotation = degreesToFirebaseRotation(rotationDegrees);
            FirebaseVisionImage FBImage = FirebaseVisionImage.fromMediaImage(mediaImage, rotation);
            Bitmap fullBitmap = FBImage.getBitmap();

            List<FirebaseVisionFace> faces = facePredictor.predict(FBImage);

            Mat fullMat = new Mat();
            Utils.bitmapToMat(fullBitmap, fullMat);
            Mat grayMat = new Mat();
            Imgproc.cvtColor(fullMat, grayMat, Imgproc.COLOR_RGBA2GRAY);

            if (faces != null) {
                ArrayList<Integer> confidences = null;
                ArrayList<opencv_core.Mat> facesMats = new ArrayList<>(faces.size());

                for (FirebaseVisionFace face : faces) {
                    opencv_core.Mat javaCvFaceMat = cropScaledJavaCvMat(grayMat, face.getBoundingBox());
                    if (javaCvFaceMat != null) facesMats.add(javaCvFaceMat);
                }

                if (!faceRecognizer.isReadyToTrain()) {
                    if (faceTracker == null) {
                        faceTracker = new FaceTracker(faces, new Size(fullBitmap.getWidth(), fullBitmap.getHeight()));
                    } else {
                        faceTracker.trackFace(faces);
                    }
                    opencv_core.Mat targetFaceMat = cropScaledJavaCvMat(grayMat, faceTracker.getTrackedFace().getBoundingBox());
                    if (targetFaceMat != null) {
                        faceRecognizer.addTargetFace(targetFaceMat);
                        setStatus("face frames left to train: " + faceRecognizer.framesLeftToTrain());
                    }
                    for (int i = 0; i < faces.size(); i++) {
                        if (faces.get(i) != faceTracker.getTrackedFace()) {
                            opencv_core.Mat faceMat = cropScaledJavaCvMat(grayMat, faces.get(i).getBoundingBox());
                            faceRecognizer.addNonTargetFace(faceMat);
                        }
                    }
                } else {
                    if (!faceRecognizer.isTrained()) {
                        setStatus("training...");

                        float thresh = faceRecognizer.train();
                        faceTracker = null;
                        setStatus("thresh was set to " + thresh);
                    }

                    confidences = new ArrayList<>(faces.size());
                    for (opencv_core.Mat javaCvFaceMat : facesMats) {
                        int probability = faceRecognizer.predictProbability(javaCvFaceMat);
                        confidences.add(probability);
                    }

                }

                for (FirebaseVisionFace face : faces) {
                    Rect faceRect = rectToCvRect(face.getBoundingBox());
                    drawRect(fullMat, faceRect, blackColor);
                }

                if (confidences != null) {
                    for (int i = 0; i < faces.size(); i++) {
                        if (confidences.get(i) > 0) {
                            Rect faceRect = rectToCvRect(faces.get(i).getBoundingBox());
                            drawRect(fullMat, faceRect, greenColor);
                        }
                        drawConfidence(fullMat, rectToCvRect(faces.get(i).getBoundingBox()), confidences.get(i));
                    }
                }

                if (faceTracker != null) {
                    drawCrossedRect(fullMat, faceTracker.getTrackedFace().getBoundingBox(), greenColor);
                }
            }

            Utils.matToBitmap(fullMat, fullBitmap);
            runOnUiThread(() -> ivBitmap.setImageBitmap(fullBitmap));
        }
    };





    private void drawConfidence(Mat image, Rect rect, int confidence) {
        int posX = (int) Math.max(rect.tl().x - 10, 0);
        int posY = (int) Math.max(rect.tl().y - 10, 0);
        Imgproc.putText(image, String.valueOf(confidence), new Point(posX, posY),
                Core.FONT_HERSHEY_PLAIN, 1, yellowColor, 2);
    }

    private opencv_core.Mat cropScaledJavaCvMat(Mat image, android.graphics.Rect rect) {
        Rect adjustedRect = adjustRectByBitmapSize(rect, image.size());
        if (adjustedRect.area() <= 0)
            return null;

        final Mat faceMat = new Mat(image, adjustedRect);
        Imgproc.resize(faceMat, faceMat, new Size(100, 100));
        return cvMat2JavaCvMat(faceMat);
    }

    private opencv_core.Mat cvMat2JavaCvMat(Mat mat) {
        return new opencv_core.Mat((Pointer) null) {{
            address = mat.getNativeObjAddr();
        }};
    }

    private void setStatus(String status) {
        runOnUiThread(() -> {
            statusView.setText(status);
        });
    }

    private void drawCrossedRect(Mat image, android.graphics.Rect rect, Scalar color) {
        drawRect(image, rectToCvRect(rect), color);
        Imgproc.line(image, new Point(rect.left, rect.top), new Point(rect.right, rect.bottom), color, 3);
        Imgproc.line(image, new Point(rect.right, rect.top), new Point(rect.left, rect.bottom), color, 3);
    }

    private void drawRect(Mat image, Rect rect, Scalar color) {
        Imgproc.rectangle(image, rect.tl(), rect.br(),
                color, 3);
    }

    private ImageAnalysis setImageAnalysis() {

        HandlerThread analyzerThread = new HandlerThread("OpenCVAnalysis");
        analyzerThread.start();

        ImageAnalysisConfig imageAnalysisConfig = new ImageAnalysisConfig.Builder()
                .setImageReaderMode(ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE)
                .setCallbackHandler(new Handler(analyzerThread.getLooper()))
                .build();

        ImageAnalysis imageAnalysis = new ImageAnalysis(imageAnalysisConfig);

        imageAnalysis.setAnalyzer(analyzer);
        return imageAnalysis;

    }

    private Rect rectToCvRect(android.graphics.Rect rect) {
        return new Rect(rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top);
    }

    private android.graphics.Rect resizeRect(android.graphics.Rect rect, Size src, Size dst) {
        double xCoeff = dst.width / src.width, yCoeff = dst.height / src.height;
        return new android.graphics.Rect((int) (rect.left * xCoeff),
                (int) (rect.top * yCoeff),
                (int) (rect.right * xCoeff),
                (int) (rect.bottom * yCoeff));
    }

    private void updateTransform() {
        Matrix mx = new Matrix();
        float w = textureView.getMeasuredWidth();
        float h = textureView.getMeasuredHeight();

        float cX = w / 2f;
        float cY = h / 2f;

        int rotationDgr;
        int rotation = (int) textureView.getRotation();

        switch (rotation) {
            case Surface.ROTATION_0:
                rotationDgr = 0;
                break;
            case Surface.ROTATION_90:
                rotationDgr = 90;
                break;
            case Surface.ROTATION_180:
                rotationDgr = 180;
                break;
            case Surface.ROTATION_270:
                rotationDgr = 270;
                break;
            default:
                return;
        }

        mx.postRotate((float) rotationDgr, cX, cY);
        textureView.setTransform(mx);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {

        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera();
            } else {
                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }

    private boolean allPermissionsGranted() {

        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }


    public static int degreesToFirebaseRotation(int degrees) {
        switch (degrees) {
            case 0:
                return FirebaseVisionImageMetadata.ROTATION_0;
            case 90:
                return FirebaseVisionImageMetadata.ROTATION_90;
            case 180:
                return FirebaseVisionImageMetadata.ROTATION_180;
            case 270:
                return FirebaseVisionImageMetadata.ROTATION_270;
            default:
                throw new IllegalArgumentException(
                        "Rotation must be 0, 90, 180, or 270.");
        }
    }

    public static Rect adjustRectByBitmapSize(android.graphics.Rect box, Size size) {
        // avoidance of going beyond bitmap size
        int width = (int)size.width;
        int height = (int)size.height;
        int left = Math.max(box.left, 0);
        int top = Math.max(box.top, 0);
        int right = Math.min(box.right, width);
        int bottom = Math.min(box.bottom, height);
        return new Rect(left, top, right - left, bottom - top);
    }

}
