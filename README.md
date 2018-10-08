# Bachelorarbeit

Erste Schritte
--------------

1. CUDA Installieren
2. OpenCV Installieren
3. Darknet installieren (Mit OpenCV und CUDA aktiviert)  
  3.1 Weightfile runterladen: `wget https://pjreddie.com/media/files/yolov3.weights`
  3.2 Darknet Testen, z.B, mit 
  `./darknet detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights <Video> `
  Falls Grafikspeicher zu gering, in yolov3.cfg batchsize und subdivision verkleinern. (z.B. beides auf 1).
 
4. Im Sourcecode das Include directory für darknet.h, sowie im Konstuktor von MatDetector die Pfade zur Config und Weightfiles    anpassen.
5. In CmakeList.txt Pfad zur Darknet Library anpassen
6. Mit Cmake erstellen
7. Mit `./Panorama2Cube <Pfad zu Videodatei> ` laufen lassen.
 
