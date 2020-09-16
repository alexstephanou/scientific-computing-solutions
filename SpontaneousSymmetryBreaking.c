#include<stdlib.h>
#include<stdio.h>
#include<math.h>

void read_input(double *L, int *N, double *t_f, double *output_timestep,double *K, double *A, double *B);

int main(void) {
    // PARAMETERS:
    
    //defining pi
    #define M_PI acos(-1.0) 
    // L, right x boundary of domain                  
    double L;
    // N, number of grid points
    int N;
    // tf, Length of time to run simulation
    double t_f;
    // output_timestep, timestep for diagnostic output.
    double output_timestep;
    //K, A and B, initial condition parameters
    double K, A, B;

    // Read in from file; 
    read_input(&L, &N, &t_f, &output_timestep, &K, &A, &B);
    
    // Grid spacing
    double dx = L/(N-1);		      
    
	// Set up time step so code is stable. Testing which of the inital conidition paramaters are largest, and then if any are >1,
    //dividing timestep by that value cubed.
    //If not, the timestep for the linearised equation is used (linearised timestep < 0.5*dx*dx)
    double dt;
	double biggestnumber = A;
	if (B>biggestnumber){
		biggestnumber = B;
	} if (K> biggestnumber){
		biggestnumber = K;
	}
	
	if (A> 1 || B>1 || K>1){
		dt = 0.25*dx*dx/(pow(biggestnumber,3));
	} else dt = 0.25*dx*dx;
	
  
  
    // GRID STORAGE:
    
    double *U, *U_next;  //U at current and next timestep
    double *V, *V_next;  //V at current and next timestep
    
    // Allocate memory according to size of N 
    U       = (double*)malloc(sizeof(double)*N);
    U_next  = (double*)malloc(sizeof(double)*N);
    V       = (double*)malloc(sizeof(double)*N);
    V_next  = (double*)malloc(sizeof(double)*N);

    if (U == NULL || U_next == NULL || V == NULL ||V_next == NULL) {
    	printf("Memory allocation failed\n");
    	return 1;
    }
  
    //initialise U and V
    int k;
    double x;
  
    for(k=0;k<N;k++) {
    	x = k*dx;
    	U[k]  = K + A*cos(2*M_PI*x/L);
    	V[k]  = B*sin(2*M_PI*x/L);
    }
  
    FILE *file;
    file = fopen("output.txt", "w"); //open output file 
  
    // Output at start of simulation excluding last point.
    double ctime = 0.0;
    for (k=0; k<N-1; k++ ) {
    	x = k*dx;
    	fprintf(file,"%g %g %g %g \n", ctime,x,U[k],V[k]);
    }
    x = (N-1)*dx;  //boundary conditions
    U[N-1]=U[0];   
    V[N-1]=V[0];
    fprintf(file,"%g %g %g %g \n", ctime,x,U[N-1],V[N-1]); //printing last point
    double next_output_time = output_timestep;

  
    //loop over timesteps 
    while (ctime<t_f){
    	double dt_new = dt;
    	int output = 0;
    	// If we would go past the next output step, reduce the timestep.
    	if (ctime+dt_new>next_output_time) {
        	dt_new = next_output_time - ctime;
        	output = 1;
    	}

    	//loop over points so that first point and second last point loop around
		//using % for km and kp makes the run time much slower, so the x=0 and x=L-dx points were looped manually
    	for (k=1; k<N-2; k++) {
			x = k*dx;
    		int km = (k-1);
			int kp = (k+1);

    		double dUdt = (U[kp] + U[km] - 2*U[k])/(dx*dx);
    		double dVdt = (V[kp] + V[km] - 2*V[k])/(dx*dx);
    		U_next[k] = U[k] + dt*dUdt + (2*U[k] - 4*pow(U[k],3) - 4*U[k]*pow(V[k],2))*dt;
    		V_next[k] = V[k] + dt*dVdt + (2*V[k] - 4*pow(V[k],3) - 4*V[k]*pow(U[k],2))*dt;
		}
		
		double dUdt_0 = (U[1] + U[N-2] - 2*U[0])/(dx*dx);
    	double dVdt_0 = (V[1] + V[N-2] - 2*V[0])/(dx*dx);
		U_next[0] = U[0] + dt*dUdt_0 + (2*U[0] - 4*pow(U[0],3) - 4*U[0]*pow(V[0],2))*dt;
		V_next[0] = V[0] + dt*dVdt_0 + (2*V[0] - 4*pow(V[0],3) - 4*V[0]*pow(U[0],2))*dt;
		
		double dUdt_last = (U[0] + U[N-3] - 2*U[N-2])/(dx*dx);
    	double dVdt_last = (V[0] + V[N-3] - 2*V[N-2])/(dx*dx);
		U_next[N-2] = U[N-2] + dt*dUdt_last + (2*U[N-2] - 4*pow(U[N-2],3) - 4*U[N-2]*pow(V[N-2],2))*dt;
		V_next[N-2] = V[N-2] + dt*dVdt_last + (2*V[N-2] - 4*pow(V[N-2],3) - 4*V[N-2]*pow(U[N-2],2))*dt;
    
	
		// Set boundary values so that first point is equal to last point ie. Z(x,t)=Z(x+L,t)
    	U_next[N-1] = U_next[0];
    	V_next[N-1] = V_next[0];
	
		// Copy next values at timestep to U,V arrays.
    	for (k=0; k<N; k++){
    		U[k] = U_next[k];
    		V[k] = V_next[k];
    	}
	

		// Increment time and print the rest of the values to the output file
    	ctime += dt_new;
    	if (output) {
    		for (k=0; k<N; k++ ) {
				x = k*dx;
				fprintf(file,"%g %g %g %g \n", ctime,x,U[k],V[k]);
    		}
    	next_output_time += output_timestep;
    	}
  }
	
	free(U);
	free(U_next);
	free(V);
	free(V_next);
	return 0;
  
	fclose(file); //close output file
}


//function to read input file
void read_input(double *L, int *N, double *t_f, double *output_timestep, double *K, double *A, double *B) {
    FILE *infile;
    if(!(infile=fopen("input.txt","r"))) {
    	printf("Error opening file\n");
    	exit(1);
    }
    if(7!=fscanf(infile,"%lf %d %lf %lf %lf %lf %lf",L,N,t_f,output_timestep,K,A,B)) {
    	printf("Error reading parameters from file\n");
    	exit(1);
    }
    fclose(infile);
}

