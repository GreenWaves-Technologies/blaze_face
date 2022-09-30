/*
 * Copyright 2021 GreenWaves Technologies, SAS
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdlib.h>
#include <math.h>
#include "face_detection_front.h"
#include "post_process.h"

/* Defines */
#define NUM_CLASSES 	2
#define AT_INPUT_SIZE 	(AT_INPUT_WIDTH*AT_INPUT_HEIGHT*AT_INPUT_COLORS)
#define PERF

#define __XSTR(__s) __STR(__s)
#define __STR(__s) #__s 

F16* scores_out;
F16* boxes_out;

AT_HYPERFLASH_FS_EXT_ADDR_TYPE face_detection_front_L3_Flash = 0;
F16 Input_1[AT_INPUT_SIZE];
char *ImageName = NULL;

static void RunNetwork()
{
#ifdef PERF
	//printf("Start timer\n");
	gap_cl_starttimer();
	gap_cl_resethwtimer();
#endif
  printf("Running on cluster\n");
  GPIO_HIGH();
  face_detection_frontCNN(Input_1,boxes_out,scores_out,&(scores_out[512]),&(boxes_out[512*16]));
  GPIO_LOW();
  printf("Runner completed\n");

}

void printBboxes_forPython(bbox_float_t *boundbxs){
	printf("\n\n======================================================");
	printf("\nThis can be copy-pasted to python to draw BoudingBoxs   ");
	printf("\n\n");

	for (int counter=0;counter< MAX_BB_OUT;counter++){
		if(boundbxs[counter].alive){
			printf("rect = patches.Rectangle((%f,%f),%f,%f,linewidth=1,edgecolor='r',facecolor='none')\nax.add_patch(rect)\n",
				boundbxs[counter].xmin,
				boundbxs[counter].ymin,
				boundbxs[counter].w,
				boundbxs[counter].h
				);
			printf("kp = patches.Circle((%f,%f),radius=1,color='green')\nax.add_patch(kp)\n",
				boundbxs[counter].k1_x,
				boundbxs[counter].k1_y);
			printf("kp = patches.Circle((%f,%f),radius=1,color='green')\nax.add_patch(kp)\n",
				boundbxs[counter].k2_x,
				boundbxs[counter].k2_y);
			printf("kp = patches.Circle((%f,%f),radius=1,color='green')\nax.add_patch(kp)\n",
				boundbxs[counter].k3_x,
				boundbxs[counter].k3_y);
			printf("kp = patches.Circle((%f,%f),radius=1,color='green')\nax.add_patch(kp)\n",
				boundbxs[counter].k4_x,
				boundbxs[counter].k4_y);
			printf("kp = patches.Circle((%f,%f),radius=1,color='green')\nax.add_patch(kp)\n",
				boundbxs[counter].k5_x,
				boundbxs[counter].k5_y);
			printf("kp = patches.Circle((%f,%f),radius=1,color='green')\nax.add_patch(kp)\n",
				boundbxs[counter].k6_x,
				boundbxs[counter].k6_y);
		}
	}//
}

float rect_intersect_area(float a_x, float a_y, float a_w, float a_h,
                         float b_x, float b_y, float b_w, float b_h ){

    #define MIN(a,b) ((a) < (b) ? (a) : (b))
    #define MAX(a,b) ((a) > (b) ? (a) : (b))

    float x = MAX(a_x,b_x);
    float y = MAX(a_y,b_y);

    float size_x = MIN(a_x+a_w,b_x+b_w) - x;
    float size_y = MIN(a_y+a_h,b_y+b_h) - y;

    if(size_x <=0 || size_x <=0)
        return 0;
    else
        return size_x*size_y;

    #undef MAX
    #undef MIN
}

void non_max_suppress(bbox_float_t * boundbxs){

    int idx,idx_int;

    //Non-max supression
     for(idx=0;idx<MAX_BB_OUT;idx++){
        //check if rect has been removed (-1)
        if(boundbxs[idx].alive==0)
            continue;

        for(idx_int=0;idx_int<MAX_BB_OUT;idx_int++){

            if(boundbxs[idx_int].alive==0 || idx_int==idx)
                continue;

            //check the intersection between rects
            float intersection = rect_intersect_area(boundbxs[idx].xmin,boundbxs[idx].ymin,boundbxs[idx].w,boundbxs[idx].h,
                                                   boundbxs[idx_int].xmin,boundbxs[idx_int].ymin,boundbxs[idx_int].w,boundbxs[idx_int].h);

            if(intersection >= NON_MAX_THRES){ //is non-max
                //supress the one that has lower score that is alway the internal index, since the input is sorted
                boundbxs[idx_int].alive =0;
            }
        }
    }
}


int checkResults(bbox_float_t *boundbxs){
    int totAliveBB=0;
    float x,y,w,h;

    for(int idx=0;idx<MAX_BB_OUT;idx++){
        if(boundbxs[idx].alive){
			totAliveBB++;
        	x = boundbxs[idx].xmin;
            y = boundbxs[idx].ymin;
            w = boundbxs[idx].w;
            h = boundbxs[idx].h;
        }
    }

    //Cabled check of result (not nice but effective) with +/- 3 px tollerance
    if(totAliveBB!=1) return -1;
    if( x > 12 + 1 || x < 12 - 1 )         return -1;
    if( y > 30 + 1 || y < 30 - 1 )         return -1;
    if( w > 46 + 1 || w < 46 - 1 )         return -1;
    if( h > 46 + 1 || h < 46 - 1 )         return -1;

    return 0;

}



int start()
{

	#ifndef __EMUL__
    	OPEN_GPIO_MEAS();
		unsigned char * ImageIn = (unsigned char *) pi_l2_malloc(sizeof(char)*(AT_INPUT_SIZE));
		boxes_out=pi_l2_malloc(sizeof(F16)*(16*896));
		scores_out=pi_l2_malloc(sizeof(F16)*(1*896));
	
		/*-----------------------OPEN THE CLUSTER--------------------------*/
		struct pi_device cluster_dev;
		struct pi_cluster_conf conf;
		pi_cluster_conf_init(&conf);
		conf.cc_stack_size = STACK_SIZE;
		pi_open_from_conf(&cluster_dev, (void *)&conf);
		pi_cluster_open(&cluster_dev);
		pi_freq_set(PI_FREQ_DOMAIN_FC, FREQ_FC*1000*1000);
		pi_freq_set(PI_FREQ_DOMAIN_CL, FREQ_CL*1000*1000);
		pi_freq_set(PI_FREQ_DOMAIN_PERIPH, FREQ_PE*1000*1000);
		printf("Set FC Frequency = %d MHz, CL Frequency = %d MHz, PERIIPH Frequency = %d MHz\n",
				pi_freq_get(PI_FREQ_DOMAIN_FC), pi_freq_get(PI_FREQ_DOMAIN_CL), pi_freq_get(PI_FREQ_DOMAIN_PERIPH));
		#ifdef VOLTAGE
		pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP, VOLTAGE);
		pi_pmu_voltage_set(PI_PMU_VOLTAGE_DOMAIN_CHIP, VOLTAGE);
		printf("Voltage: %dmV\n", VOLTAGE);
		#endif
	#else
		Output_1=malloc(sizeof(char)*(16*16*32+96*8*8));
		Output_2=malloc(sizeof(char)*(16*16*2+8*8*6));
	#endif

	printf("Reading image %s\n", ImageName);
	//Reading Image from Bridge
	if (ReadImageFromFile(ImageName, AT_INPUT_WIDTH, AT_INPUT_HEIGHT, AT_INPUT_COLORS, ImageIn, AT_INPUT_SIZE*sizeof(char), IMGIO_OUTPUT_CHAR, 0)) {
	printf("Failed to load image %s\n", ImageName);
	return 1;
	}
	printf("Constructing Network\n");
	int err_construct = face_detection_frontCNN_Construct();
	if (err_construct){
		printf("Graph Constructor exited with error: %d\n", err_construct);
		return -1;
	}
	printf("Graph constructor was OK\n");

	for (int h=0; h<AT_INPUT_HEIGHT; h++) {
		for (int w=0; w<AT_INPUT_WIDTH; w++) {
			for (int c=0; c<AT_INPUT_COLORS; c++) {
				Input_1[c*AT_INPUT_WIDTH*AT_INPUT_HEIGHT+h*AT_INPUT_WIDTH+w] = (((F16) ImageIn[h*AT_INPUT_WIDTH*AT_INPUT_COLORS+w*AT_INPUT_COLORS+c]) / 128) - 1.0f;
			}
		}
	}

	#ifndef __EMUL__
		/*--------------------------TASK SETUP------------------------------*/
		struct pi_cluster_task *task_encoder = pi_l2_malloc(sizeof(struct pi_cluster_task));
		if(task_encoder==NULL) {
			printf("pi_cluster_task alloc Error!\n");
			pmsis_exit(-1);
		}
		printf("Stack size is %d and %d\n",STACK_SIZE,SLAVE_STACK_SIZE );
		pi_cluster_task(task_encoder, &RunNetwork, NULL);
		pi_cluster_task_stacks(task_encoder, NULL, SLAVE_STACK_SIZE);
		pi_cluster_send_task_to_cl(&cluster_dev, task_encoder);
	#else
		RunNetwork();
	#endif

	#ifdef PERF
	{
		unsigned int TotalCycles = 0, TotalOper = 0;
		printf("\n");
		for (int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
			printf("%45s: Cycles: %10d, Operations: %10d, Operations/Cycle: %f\n", AT_GraphNodeNames[i],
			       AT_GraphPerf[i], AT_GraphOperInfosNames[i], ((float) AT_GraphOperInfosNames[i])/ AT_GraphPerf[i]);
			TotalCycles += AT_GraphPerf[i]; TotalOper += AT_GraphOperInfosNames[i];
		}
		printf("\n");
		printf("%45s: Cycles: %10d, Operations: %10d, Operations/Cycle: %f\n", "Total", TotalCycles, TotalOper, ((float) TotalOper)/ TotalCycles);
		printf("\n");
	}
	#endif

    printf("Destructing Network\n");
	face_detection_frontCNN_Destruct();
	
// /* ------------------------------------------------------------------------- */

	float *scores = pi_l2_malloc(896*sizeof(float));
	float *boxes  = pi_l2_malloc(16*896*sizeof(float));
	bbox_float_t* bboxes = pi_l2_malloc(MAX_BB_OUT*sizeof(bbox_float_t));

	if(scores==NULL || boxes==NULL || bboxes==NULL){
		printf("Alloc error\n");
		pmsis_exit(-1);
	}
	printf("\n");
	for(int i=0;i<896;i++){
		if(i<512)
			scores[i] = 1/(1+exp(-(((float)scores_out[i]))));
		else
			scores[i] = 1/(1+exp(-(((float)scores_out[i]))));

		for(int j=0;j<16;j++){
			if(i<512)
				boxes[(i*16)+j] = ((float)boxes_out[(i*16)+j]);
			else
				boxes[(i*16)+j] = ((float)boxes_out[(i*16)+j]);
		}	
	}

  	post_process(scores,boxes,bboxes,128,128, 0.5f);

  	non_max_suppress(bboxes);
  	printBboxes_forPython(bboxes);

  	for(int i=0;i<MAX_BB_OUT;i++){
  		if (bboxes[i].alive)
  			printf("%f %f %f %f %f\n",bboxes[i].score, bboxes[i].xmin,bboxes[i].ymin,bboxes[i].w,bboxes[i].h);
  	}

  	pi_l2_free(scores,896*sizeof(float));
  	pi_l2_free(boxes,16*896*sizeof(float));

	if(checkResults(bboxes)){
		printf("Output is not correct...\n");
		pmsis_exit(-1);
	}else{
		printf("Output correct!\n");
	}

	pi_l2_free(bboxes,MAX_BB_OUT*sizeof(bbox_t));

	#ifndef __EMUL__
		pi_l2_free(scores_out,sizeof(F16)*(1*896));
		pi_l2_free(boxes_out,sizeof(F16)*(16*896));

		pmsis_exit(0);
	#else
		free(Output_2);
		free(Output_1);
	#endif

	printf("Ended\n");
	return 0;
}

int main(int argc, char *argv[])
{
    #ifndef __EMUL__
    ImageName = __XSTR(AT_IMAGE);
    printf("\n\n\t *** NNTOOL BlazeFace int8 ***\n\n");
    return pmsis_kickoff((void *) start);
    #else
    if (argc < 2)
    {
        printf("Usage: ./exe [image_file]\n");
        exit(-1);
    }
    ImageName = argv[1];
    printf("\n\n\t *** NNTOOL BlazeFace int8 ***\n\n");
    start(NULL);
    return 0;
    #endif
}
