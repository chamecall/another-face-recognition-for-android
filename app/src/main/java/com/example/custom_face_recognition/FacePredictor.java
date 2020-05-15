package com.example.custom_face_recognition;

import com.google.android.gms.tasks.Task;
import com.google.android.gms.tasks.Tasks;
import com.google.firebase.ml.vision.FirebaseVision;
import com.google.firebase.ml.vision.common.FirebaseVisionImage;
import com.google.firebase.ml.vision.face.FirebaseVisionFace;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetector;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetectorOptions;

import java.util.List;
import java.util.concurrent.ExecutionException;

public class FacePredictor {
    private FirebaseVisionFaceDetectorOptions options;


    public FacePredictor() {
        this.options = new FirebaseVisionFaceDetectorOptions.Builder()
                .setContourMode(FirebaseVisionFaceDetectorOptions.NO_CONTOURS)
                .setLandmarkMode(FirebaseVisionFaceDetectorOptions.NO_LANDMARKS)
                .setPerformanceMode(FirebaseVisionFaceDetectorOptions.FAST)
                .build();
    }

    public List<FirebaseVisionFace> predict(FirebaseVisionImage FBImage) {

        FirebaseVisionFaceDetector faceDetector = FirebaseVision.getInstance().getVisionFaceDetector(options);

        List<FirebaseVisionFace> rets = null;

        Task<List<FirebaseVisionFace>> task = faceDetector.detectInImage(FBImage);
        try {
            List<FirebaseVisionFace> results = Tasks.await(task);
            if (!results.isEmpty()) {
                rets = results;
            }
        } catch (ExecutionException | InterruptedException e) {
            e.printStackTrace();
        }

        return rets;
    }


}
