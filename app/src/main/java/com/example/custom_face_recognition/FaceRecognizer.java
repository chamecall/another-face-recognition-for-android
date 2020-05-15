package com.example.custom_face_recognition;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_face;

import java.nio.IntBuffer;
import java.util.ArrayList;

import static org.opencv.core.CvType.CV_32SC1;

public class FaceRecognizer {
    private int framesToTrain, framesToTest;
    private opencv_face.FaceRecognizer mLBPHFaceRecognizer;
    private float lbphThresh;
    private final float CONF_THRESH_VAR_COEFF = 1.5f;

    private ArrayList<opencv_core.Mat> capturedTargetFaces;
    private ArrayList<opencv_core.Mat> capturedNonTargetFaces;
    private float avgTargetConf;

    FaceRecognizer() {
        this.framesToTrain = 10;
        this.framesToTest = 5;
        capturedTargetFaces = new ArrayList<>(framesToTrain + framesToTest);
        capturedNonTargetFaces = new ArrayList<>();
    }

    boolean addTargetFace(opencv_core.Mat face) {
        return capturedTargetFaces.add(face);
    }

    boolean addNonTargetFace(opencv_core.Mat face) {
        return capturedNonTargetFaces.add(face);
    }

    boolean isReadyToTrain() {
        return capturedTargetFaces.size() == framesToTrain + framesToTest;
    }

    int framesLeftToTrain() {
        return framesToTrain + framesToTest - capturedTargetFaces.size();
    }

    boolean isTrained() {
        return mLBPHFaceRecognizer != null;
    }

    float train() {
        opencv_core.Mat labels = new opencv_core.Mat(framesToTrain, 1, CV_32SC1);
        opencv_core.MatVector faceImages = new opencv_core.MatVector(framesToTrain);

        mLBPHFaceRecognizer = opencv_face.LBPHFaceRecognizer.create();
        IntBuffer intBuffer = labels.createBuffer();

        for (int i = 0; i < faceImages.size(); i++) {
            faceImages.put(i, capturedTargetFaces.get(i));
            intBuffer.put(i, 1);
        }
        mLBPHFaceRecognizer.train(faceImages, labels);

        float[] confs = findConfThresh();
        lbphThresh = confs[0];
        avgTargetConf = confs[1];
        return lbphThresh;
    }

    private float[] findConfThresh() {
        float confThresh;
        float avgTargetConf = getAvgConf(capturedTargetFaces, framesToTrain, framesToTest);
        if (capturedNonTargetFaces.size() > 0) {
            float avgNonTargetConf = getAvgConf(capturedNonTargetFaces);
            confThresh = (avgTargetConf + avgNonTargetConf) / 2;
        } else {
            confThresh = avgTargetConf * CONF_THRESH_VAR_COEFF;
        }

        return new float[]{confThresh, avgTargetConf};
    }

    float getAvgConf(ArrayList<opencv_core.Mat> faces, int shift, int count) {
        float confSum = 0, avgConf;
        for (int i = shift; i < shift + count; i++) {
            float conf = predictConfidence(faces.get(i));
            confSum += conf;
        }
        avgConf = confSum / count;
        return avgConf;
    }

    float getAvgConf(ArrayList<opencv_core.Mat> faces) {
        return getAvgConf(faces, 0, faces.size());
    }

    float predictConfidence(opencv_core.Mat faceMat) {
        int[] label = new int[1];
        double[] confidence = new double[1];
        mLBPHFaceRecognizer.predict(faceMat, label, confidence);
        return (float) confidence[0];
    }

    int predictProbability(opencv_core.Mat faceMat) {
        int probability = 0;
        float conf = predictConfidence(faceMat);
        if (conf < avgTargetConf) {
            probability = 100;
        } else if (conf <= lbphThresh) {
            probability = (int) ((1 - (conf - avgTargetConf) / (lbphThresh - avgTargetConf)) * 100);
        }
        return probability;
    }


}
