package com.example.custom_face_recognition;

import com.google.firebase.ml.vision.face.FirebaseVisionFace;

import org.opencv.core.Point;
import org.opencv.core.Size;

import java.util.List;

public class FaceTracker {
    private FirebaseVisionFace trackedFace;

    FaceTracker (List<FirebaseVisionFace> faces, Size imageSize) {
        trackedFace = getNearestFace(new Point(imageSize.width / 2.0, imageSize.height / 2.0), faces);
    }

    public FirebaseVisionFace getTrackedFace() {
        return trackedFace;
    }

    void trackFace(List<FirebaseVisionFace> faces) {
        trackedFace = getNearestFace(getCenterPosFromFaceRect(trackedFace), faces);
    }

    private FirebaseVisionFace getNearestFace(Point facePos2Find, List<FirebaseVisionFace> faces) {
        FirebaseVisionFace face = null;
        // just big value
        double smallestDist2PrevFace = 9999;
        for (int i = 0; i < faces.size(); i++) {
            Point facePos = getCenterPosFromFaceRect(faces.get(i));
            double dist2PrevFace = getDistBtwPoints(facePos2Find, facePos);
            if (face == null || dist2PrevFace < smallestDist2PrevFace) {
                face = faces.get(i);
                smallestDist2PrevFace = dist2PrevFace;
            }
        }
        return face;
    }

    private Point getCenterPosFromFaceRect(FirebaseVisionFace face) {
        return new Point(face.getBoundingBox().exactCenterX(), face.getBoundingBox().exactCenterY());
    }

    public static double getDistBtwPoints(Point fPoint, Point sPoint) {
        return Math.sqrt(Math.pow(fPoint.x - sPoint.x, 2) + Math.pow(fPoint.y - sPoint.y, 2));
    }


}
