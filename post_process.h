
#ifndef __POST_PROCESS_H__
#define __POST_PROCESS_H__

#include "pmsis.h"

#define MAX_BB_OUT 15

#ifdef POST_PROCESS_OUTPUT_REVERSED

#define box_offset_y 		1
#define box_offset_x 		0
#define box_offset_height 	3
#define box_offset_width 	2
#define keypoint_offset_y 	1
#define keypoint_offset_x 	0

#else

#define box_offset_y 		0
#define box_offset_x 		1
#define box_offset_height 	2
#define box_offset_width 	3
#define keypoint_offset_y 	0
#define keypoint_offset_x 	1

#endif

#define Y_SCALE 128
#define X_SCALE 128
#define W_SCALE 128
#define H_SCALE 128


typedef struct 
{
	int    xmin;
	int    ymin;
	int    xmax;
	int    ymax;
	float score;
}bbox_t;

int post_process(float* scores,float * boxes,bbox_t* bboxes,int width,int height, float thres);

#endif //__POST_PROCESS_H__