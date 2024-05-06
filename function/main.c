#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* PMSIS includes */
#include "pmsis.h"

//define neural network parameters
#define INPUT_SIZE 6
#define OUTPUT_SIZE 1
#define LEARNING_RATE 0.026
#define EPOCHS 15000

#define RAN_NUM  3883838
#define RAN_MAX  RAN_NUM + 1378

void cluster_helloworld(void *arg);
void cluster_delegate(void *arg);
void helloworld(void);
double activation(double x);
void initialize(void);
double predict(double inputs[]);
void train(double inputs[], double target);
void make_NeuralTest_data(void);
void train_NeuralNetwork(void);
void prediction_NeuralNetwork(void);
void performance_analysis(void);


double training_data[][INPUT_SIZE] = {

{1.96, 1.95, 1.95, 1.94, 1.94, 1.94, 1.93, 1.96, 1.94, 1.94, 1.93, 1.93, 1.94, 1.94, 1.95, 1.95, 1.95, 1.94, 1.93, 1.93, 1.94, 1.95, 1.95, 1.95, 1.94, 1.94, 1.94, 1.93, 1.94, 1.95, 1.95, 1.93, 1.94, 1.94, 1.94, 1.94, 1.96, 1.95, 1.96, 1.94, 1.94, 1.94, 1.94, 1.94, 1.95, 1.95, 1.96, 1.94, 1.95, 1.94, 1.94, 1.94, 1.94, 1.96, 1.96, 1.95, 1.94, 1.94, 1.94},

{1.83, 1.85, 1.85, 1.88, 1.86, 1.87, 1.83, 1.85, 1.85, 1.86, 1.86, 1.88, 1.85, 1.87, 1.88, 1.84, 1.86, 1.87, 1.84, 1.90, 1.90, 1.88, 1.88, 1.89, 1.89, 1.86, 1.91, 1.89, 1.92, 1.87, 1.87, 1.90, 1.86, 1.84, 1.89, 1.88, 1.87, 1.89, 1.89, 1.86, 1.92, 1.84, 1.87, 1.85, 1.85, 1.82, 1.80, 1.79, 1.81, 1.82, 1.86, 1.85, 1.82, 1.80, 1.81, 1.82, 1.79, 1.84, 1.84}, 

{1.74, 1.75, 1.75, 1.81, 1.78, 1.74, 1.78, 1.75, 1.77, 1.74, 1.73, 1.77, 1.75, 1.77, 1.77, 1.77, 1.75, 1.81, 1.81, 1.77, 1.74, 1.79, 1.73, 1.76, 1.78, 1.77, 1.82, 1.79, 1.77, 1.78, 1.77, 1.77, 1.74, 1.78, 1.78, 1.78, 1.79, 1.80, 1.76, 1.75, 1.82, 1.79, 1.76, 1.78, 1.79, 1.77, 1.75, 1.74, 1.78, 1.78, 1.78, 1.78, 1.78, 1.76, 1.82, 1.79, 1.81, 1.81, 1.78},

{1.60, 1.63, 1.63, 1.61, 1.68, 1.66, 1.66, 1.66, 1.64, 1.63, 1.60, 1.60, 1.62, 1.66, 1.64, 1.65, 1.63, 1.65, 1.60, 1.64, 1.63, 1.62, 1.64, 1.62, 1.62, 1.60, 1.59, 1.59, 1.63, 1.62, 1.61, 1.60, 1.63, 1.65, 1.59, 1.61, 1.65, 1.64, 1.60, 1.63, 1.61, 1.59, 1.63, 1.60, 1.60, 1.60, 1.60, 1.59, 1.60, 1.59, 1.58, 1.61, 1.62, 1.61, 1.59, 1.58, 1.61, 1.59, 1.59},

{1.55, 1.53, 1.53, 1.50, 1.53, 1.54, 1.53, 1.57, 1.55, 1.54, 1.55, 1.52, 1.51, 1.55, 1.57, 1.57, 1.56, 1.55, 1.55, 1.55, 1.53, 1.55, 1.54, 1.55, 1.53, 1.55, 1.53, 1.53, 1.52, 1.52, 1.57, 1.55, 1.56, 1.56, 1.51, 1.55, 1.57, 1.57, 1.56, 1.55, 1.57, 1.57, 1.55, 1.57, 1.59, 1.59, 1.55, 1.55, 1.56, 1.56, 1.58, 1.56, 1.59, 1.55, 1.55, 1.57, 1.56, 1.54, 1.54},

{1.41, 1.38, 1.40, 1.41, 1.40, 1.37, 1.39, 1.40, 1.41, 1.40, 1.40, 1.38, 1.39, 1.44, 1.44, 1.40, 1.41, 1.39, 1.39, 1.40, 1.39, 1.44, 1.40, 1.43, 1.41, 1.43, 1.47, 1.43, 1.38, 1.44, 1.40, 1.39, 1.39, 1.41, 1.41, 1.40, 1.37, 1.44, 1.43, 1.44, 1.39, 1.43, 1.47, 1.42, 1.44, 1.48, 1.47, 1.43, 1.43, 1.42, 1.44, 1.43, 1.44, 1.44, 1.49, 1.46, 1.44, 1.43, 1.44}

};

double test_data[][INPUT_SIZE] = {};

double targets[] = {0,20,40,60,80,90};


void performance_analysis(void)
{ 

	uint32_t tim_cycles = pi_perf_read(PI_PERF_CYCLES); 

	uint32_t tim_actcycles = pi_perf_read(PI_PERF_ACTIVE_CYCLES); 

	uint32_t tim_instr = pi_perf_read(PI_PERF_INSTR); 

	uint32_t tim_ldstall = pi_perf_read(PI_PERF_LD_STALL); 

	uint32_t tim_jrstall = pi_perf_read(PI_PERF_JR_STALL); 

	uint32_t tim_imiss = pi_perf_read(PI_PERF_IMISS); 

	uint32_t tim_ld = pi_perf_read(PI_PERF_LD); 

	uint32_t tim_st = pi_perf_read(PI_PERF_ST); 

	uint32_t tim_jump = pi_perf_read(PI_PERF_JUMP); 

	uint32_t tim_branch = pi_perf_read(PI_PERF_BRANCH); 

	uint32_t tim_btaken = pi_perf_read(PI_PERF_BTAKEN); 

	uint32_t tim_rvc = pi_perf_read(PI_PERF_RVC); 

	uint32_t tim_ldext = pi_perf_read(PI_PERF_LD_EXT); 

	uint32_t tim_stext = pi_perf_read(PI_PERF_ST_EXT); 

	uint32_t tim_ldextcyc = pi_perf_read(PI_PERF_LD_EXT_CYC); 

	uint32_t tim_stextcyc = pi_perf_read(PI_PERF_ST_EXT_CYC); 

	uint32_t tim_tcdmcont = pi_perf_read(PI_PERF_TCDM_CONT); 

	 

	printf("Total number of cycles :  %d\n" , tim_cycles); 

	printf("Total number of active cycles :  %d\n" , tim_actcycles); 

	printf("Total number of instruction :  %d\n", tim_instr ); 

	printf("Total number of load data hazard :  %d\n", tim_ldstall); 

	printf("Total number of jump register data hazard :  %d\n", tim_jrstall); 

	printf("Total number of cycles wating for inst fetch :  %d\n", tim_imiss  ); 

	printf("Total number of data memory loads executed :  %d\n" , tim_ld); 

	printf("Total number of memory stores :  %d\n" , tim_st); 

	printf("Total number of unconditional jumps :  %d\n", tim_jump); 

	printf("Total number of branches :  %d\n", tim_branch); 

	printf("Total number of taken branches :  %d\n" , tim_btaken); 

	printf("Total number of compressed inst executed :  %d\n" , tim_rvc); 

	printf("Total number of memory load to ext executed :  %d\n" , tim_ldext); 

	printf("Total number of memory stores to ext executed :  %d\n" , tim_stext); 

	printf("Total number of cycle used for memory loads to ext :  %d\n" , tim_ldextcyc); 

	printf("Total number of cycle used for memory stores to ext :  %d\n" , tim_stextcyc); 

	printf("Total number of cycle wasted due to tcdm :  %d\n" , tim_tcdmcont); 

 

} 


/* Task executed by cluster cores. */
void cluster_helloworld(void *arg)
{
    uint32_t core_id = pi_core_id(), cluster_id = pi_cluster_id();
    if( (core_id == 1) )
    {
    	initialize();
    	printf("cluster_id : %d | core : %d\n", cluster_id, core_id);
    	
    }
    else if( (core_id == 2)  )
    {
    	train_NeuralNetwork();
    	prediction_NeuralNetwork();
    	printf("cluster_id : %d | core : %d\n", cluster_id, core_id);
    	
    }
    else if( (core_id == 3)  )
    {
    	rt_team_barrier(); // wait fot other core to finish
    	prediction_NeuralNetwork();
    	printf("cluster_id : %d | core : %d\n", cluster_id, core_id);
    }
    
}



/* Cluster main entry, executed by core 0. */

void cluster_delegate(void *arg)
{
    printf("Cluster master core entry and Total number of cores : %d\n",pi_cl_cluster_nb_cores());
    /* Task dispatch to cluster cores. */
    pi_cl_team_fork(pi_cl_cluster_nb_cores(), cluster_helloworld, arg);
    printf("Cluster master core exit\n");
}

void helloworld(void)
{
    printf("Entering main controller\n");

    uint32_t errors = 0;
    uint32_t core_id = pi_core_id(), cluster_id = pi_cluster_id();
    printf("[%d %d] Hello World main!\n", cluster_id, core_id);

    struct pi_device cluster_dev;
    struct pi_cluster_conf cl_conf;

    /* Init cluster configuration structure. */
    pi_cluster_conf_init(&cl_conf);
    cl_conf.id = 0;                /* Set cluster ID. */
    /* Configure & open cluster. */
    pi_open_from_conf(&cluster_dev, &cl_conf);

    pi_perf_conf(1<<PI_PERF_CYCLES|1<<PI_PERF_ACTIVE_CYCLES|1<<PI_PERF_INSTR| 1<<PI_PERF_LD_STALL|1<<PI_PERF_JR_STALL|1<<PI_PERF_IMISS|1<<PI_PERF_LD|1<<PI_PERF_ST|1<<PI_PERF_JUMP|1<<PI_PERF_BRANCH|1<<PI_PERF_BTAKEN|1<<PI_PERF_RVC|1<<PI_PERF_LD_EXT|1<<PI_PERF_ST_EXT|1<<PI_PERF_LD_EXT_CYC|1<<PI_PERF_ST_EXT_CYC|1<<PI_PERF_TCDM_CONT);
    pi_perf_reset();
    pi_perf_start();

    if (pi_cluster_open(&cluster_dev))
    {
        printf("Cluster open failed !\n");
        pmsis_exit(-1);
    }

    pi_perf_stop();
    performance_analysis();
    

    /* Prepare cluster task and send it to cluster. */
    struct pi_cluster_task cl_task;

    pi_cluster_send_task_to_cl(&cluster_dev, pi_cluster_task(&cl_task, cluster_delegate, NULL));

    pi_cluster_close(&cluster_dev);

    printf("Test success !\n");

    pmsis_exit(errors);
}
 
//define ReLU activated function

double activation(double x){
    return(x>0)?x:0.0;
}
 
//define neural network weights and bias
double weights[INPUT_SIZE];
double bias;
 
//initialize NN parameters

void initialize(void)
{
int ran_inc = 0;
    for(int i=0; i<INPUT_SIZE;i++)
    {
        weights[i] = ( ((double) (RAN_NUM + ran_inc)/RAN_MAX) * 2-1) * sqrt(2.0 / INPUT_SIZE); //initialize random weights
	ran_inc++;
    }
    bias = ((double)(RAN_NUM + ran_inc)/RAN_MAX) * 2-1; //initialize random bias
	ran_inc++;
}
 
//neural networks forward propgation

double predict(double inputs[])
{
    double output = 0;
    for(int i=0;i<INPUT_SIZE;i++)
    {
        output += weights[i] * inputs[i];
    }
    output +=  bias;
    return activation(output);
}
 
//Train a neural network

void train(double inputs[], double target)
{
    double prediction = predict(inputs);
    double error = target - prediction;
    for(int i=0;i<INPUT_SIZE;i++){
        weights[i]+= LEARNING_RATE* error * inputs[i];
    }
    bias += LEARNING_RATE* error;
}

void make_NeuralTest_data(void)
{
	// Test neural network
	memcpy(test_data,training_data,sizeof(training_data));
    	
	for(int j = 0; j < INPUT_SIZE;j++)
	{
		for(int i = 0; i < sizeof(test_data)/sizeof(test_data[0]);i++)
		{
			if( i % 2 )
			{
				test_data[i][j] += 0.001;
			}
			else
			{
				test_data[i][j] += 0.001;
			}
		}
	}
}

void train_NeuralNetwork(void)
{
	for(int epoch = 0;epoch<EPOCHS;epoch++)
	{
		for(int i=0;i<sizeof(training_data)/sizeof(training_data[0]);i++)
		{
		    train(training_data[i], targets[i]);
		}
	}
}

void prediction_NeuralNetwork(void)
{
	make_NeuralTest_data();

    	for(int i=0;i<sizeof(training_data)/sizeof(training_data[0]);i++)
    	{
        	double prediction = predict(test_data[i]);
        	printf("INPUT:[%lf,%lf], Target: %lf, Prediction: %lf\n",training_data[i][0], training_data[i][1],targets[i],prediction);
    	}
}
 
int main(void)
{

	//initialize the neural networks
	//initialize();
	//prepare training data
	//train_NeuralNetwork();
	//prepare test data and perform prediction
	//prediction_NeuralNetwork();
	//Performance analyze
	


	printf("\n\n\t *** PMSIS HelloWorld ***\n\n");
	return pmsis_kickoff((void *) helloworld);

	return 0;
}
