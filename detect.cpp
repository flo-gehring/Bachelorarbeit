#include "detect.h"//
// Created by flo on 06.10.18.
//

#include "detect.h"
#include <string.h>
#include <opencv2/dnn.hpp>
#include <vector>
MatDetector::MatDetector(){

    fps = 0;
    demo_thresh = 0.5f;
    demo_hier = 0.5f;
    running = 0;
    demo_frame = 3;
    demo_index = 0;
    demo_done = 0;
    demo_total = 0;





    cfgfile = "/home/flo/Workspace/darknet/cfg/yolov3.cfg";
    weightfile = "/home/flo/Workspace/darknet/yolov3.weights";
    datacfg = "/home/flo/Workspace/darknet/cfg/coco.data";

    image **alphabet = load_alphabet();
    net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);

    list *options = read_data_cfg(datacfg);
    int classes = option_find_int(options, "classes", 20);
    char *name_list = option_find_str(options, "names", "/home/flo/Workspace/darknet/names.list");
    char **names = get_labels(name_list);



    predictions = reinterpret_cast<float**>(calloc(1, sizeof(float*)));
    predictions[0] = reinterpret_cast<float*>(calloc(size_network(net), sizeof(float)));
    avg = reinterpret_cast<float *>(calloc(size_network(net), sizeof(float)));

    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_hier = hier;



}

void MatDetector::detect_and_display(cv::Mat input_mat){

    found.clear();

    //Convert Mat to the "image" Format which darknet uses.
    IplImage iplimage = input_mat;
    image darknet_image = ipl_to_image(&iplimage);
    rgbgr_image(darknet_image);
    image letterbox = letterbox_image(darknet_image, net->w, net->h);


    running = 1;
    float nms = .95;

    layer l = net->layers[net->n-1];
    float *X = letterbox.data;
    double time = what_time_is_it_now();
    network_predict(net, X);
    printf("%s: Predicted in %f seconds.\n", "Mat", what_time_is_it_now()-time);


    //    remember_network(net);
    detection *dets = 0;
    int nboxes = 0;
    //dets = avg_predictions(net, &nboxes);
    dets = get_network_boxes(net, darknet_image.w, darknet_image.h, thresh, demo_thresh, 0, 1, &nboxes);

    if (nms > 0) do_nms_obj(dets, nboxes, l.classes, nms);


    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n",fps);
    printf("Objects:\n\n");
    image display = darknet_image;
    // draw_detections(display, dets, nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes);
    print_detections(display, dets, nboxes);
    free_detections(dets, nboxes);
    free_image(darknet_image); // Free images against memory leakage
    free_image(letterbox);



}

detection * MatDetector::avg_predictions(network *net, int *nboxes)
{
    int i, j;
    int count = 0;
    fill_cpu(size_network(net), 0, avg, 1);

    axpy_cpu(demo_total, 1./demo_frame, predictions[0], 1, avg, 1);

    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            memcpy(l.output, avg + count, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
    detection *dets = get_network_boxes(net, darknet_image.w,
            darknet_image.h, demo_thresh, demo_hier, 0, 1, nboxes);
    return dets;
}


int size_network(network *net)
{
    int i;
    int count = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            count += l.outputs;
        }
    }
    return count;
}


void  MatDetector::print_detections(image im, detection *dets, int num){

    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> indices;

    // Gleiches vorgehen wie in draw_detections
    int i,j;

    for(i = 0; i < num; ++i) {
        float prob = 0;
        char labelstr[4096] = {0};
        int classno = -1;
        for (j = 0; j < classes; ++j) {
            if (dets[i].prob[j] > thresh) {
                if (classno < 0) {
                    strcat(labelstr, demo_names[j]);
                    classno = j;
                } else {
                    strcat(labelstr, ", ");
                    strcat(labelstr, demo_names[j]);
                }
                printf("%s: %.0f%%\n", demo_names[j], dets[i].prob[j] * 100);
                prob = dets[i].prob[j] * 100;
            }
        }
        if (classno >= 0) { // A class was detected with sufficient threshold
            box bounding_box = dets[i].bbox;

            // Bounding Box values are given in fractions in relation
            // to the image!
            int left  = int((bounding_box.x-bounding_box.w/2.)*im.w);
            int right = int((bounding_box.x+bounding_box.w/2.)*im.w);
            int top   = int((bounding_box.y-bounding_box.h/2.)*im.h);
            int bot   = int((bounding_box.y+bounding_box.h/2.)*im.h);

            if(left < 0) left = 0;
            if(right > im.w-1) right = im.w-1;
            if(top < 0) top = 0;
            if(bot > im.h-1) bot = im.h-1;
            printf("%s: %f \n Left/Right/Top/Bot: %i/%i/%i/%i", labelstr,
                    prob, left, right, top, bot);

            bboxes.push_back(cv::Rect(cv::Point(left, top),
                    cv::Point(right, bot)));
            scores.push_back(prob);

        }
    }

    /*
     * Perform Non Maximum supression on the Bounding Boxes
     * TODO: Replace class_name placeholder with correct Class
     * TODO: Make nms threshold and confidence Threhsold a variable.
     */
    cv::dnn::NMSBoxes(bboxes, scores, .5, 0.5, indices);
    for(auto it = indices.begin(); it != indices.end(); ++it){

        found.push_back(AbsoluteBoundingBoxes());
        //AbsoluteBoundingBoxes  abs = found.front();

        found.back().rect = bboxes[*it];
        found.back().class_name = "KONSTANT PERSON";
        found.back().prob = scores[*it];
    }
}

void MatDetector::remember_network(network *net)
{
    int i;
    int count = 0;
    for(i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
            memcpy(predictions[demo_index] + count, net->layers[i].output, sizeof(float) * l.outputs);
            count += l.outputs;
        }
    }
}


image ipl_to_image(IplImage* src)
{
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    image im = make_image(w, h, c);
    unsigned char *data = (unsigned char *)src->imageData;
    int step = src->widthStep;
    int i, j, k;

    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){
                im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
            }
        }
    }
    return im;
}
