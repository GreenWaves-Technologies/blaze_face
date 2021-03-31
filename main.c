/*
 * Copyright 2019 GreenWaves Technologies, SAS
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

#include "face_detection_front.h"


/* Defines */
#define NUM_CLASSES 	2
#define AT_INPUT_SIZE 	(AT_INPUT_WIDTH*AT_INPUT_HEIGHT*AT_INPUT_COLORS)

#define __XSTR(__s) __STR(__s)
#define __STR(__s) #__s 

typedef signed char NETWORK_OUT_TYPE;


signed char Output_1[16*16*32+96*8*8];
signed char Output_2[16*16*2+8*8*6];
AT_HYPERFLASH_FS_EXT_ADDR_TYPE face_detection_front_L3_Flash = 0;
signed char Input_1[AT_INPUT_SIZE];
char *ImageName = NULL;


static void RunNetwork()
{
#ifdef PERF
	//printf("Start timer\n");
	gap_cl_starttimer();
	gap_cl_resethwtimer();
#endif
  printf("Running on cluster\n");
  face_detection_frontCNN(Input_1, Output_1, Output_2);
  printf("Runner completed\n");
  printf("Output_1 = %d\n", Output_1[0]);
  printf("Output_2 = %d\n", Output_2[0]);

}

int start()
{
	#ifndef __EMUL__
		/*-----------------------OPEN THE CLUSTER--------------------------*/
		struct pi_device cluster_dev;
		struct pi_cluster_conf conf;
		pi_cluster_conf_init(&conf);
		pi_open_from_conf(&cluster_dev, (void *)&conf);
		pi_cluster_open(&cluster_dev);
	#endif

	printf("Reading image %s\n", ImageName);
	//Reading Image from Bridge
	if (ReadImageFromFile(ImageName, AT_INPUT_WIDTH, AT_INPUT_HEIGHT, AT_INPUT_COLORS, Input_1, AT_INPUT_SIZE*sizeof(char), IMGIO_OUTPUT_CHAR, 0)) {
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

	#ifndef __EMUL__
		/*--------------------------TASK SETUP------------------------------*/
		struct pi_cluster_task *task_encoder = pmsis_l2_malloc(sizeof(struct pi_cluster_task));
		if(task_encoder==NULL) {
			printf("pi_cluster_task alloc Error!\n");
			pmsis_exit(-1);
		}
		printf("Stack size is %d and %d\n",STACK_SIZE,SLAVE_STACK_SIZE );
		memset(task_encoder, 0, sizeof(struct pi_cluster_task));
		task_encoder->entry = &RunNetwork;
		task_encoder->stack_size = STACK_SIZE;
		task_encoder->slave_stack_size = SLAVE_STACK_SIZE;
		task_encoder->arg = NULL;
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
	printf("Encoder destructed\n\n");

/* ------------------------------------------------------------------------- */
	#ifndef __EMUL__
		pmsis_exit(0);
	#endif
	printf("Ended\n");
	return 0;
}

#ifdef __EMUL__
void main(int argc, char *argv[])
{
	if (argc < 2) {
	printf("Usage: %s [image_file]\n", argv[0]);
	exit(1);
	}
	ImageName = argv[1];
	start(NULL);
}
#else
void main()
{

#define __XSTR(__s) __STR(__s)
#define __STR(__s) #__s
    printf("\n\n\t *** NNTOOL MAIN APPL ***\n\n");
    ImageName = __XSTR(AT_IMAGE);
   	pmsis_kickoff((void *)start);
}
#endif
