#include <stdlib.h>
#include <stdio.h>
#include <lapacke.h>
#include <math.h>

void read_inputs(double *L, int *N, double *K, double *gamma, double *omega_a, double *omega_b, int *z);

struct band_mat{
	long ncol;        /* Number of columns in band matrix            */
	long nbrows;      /* Number of rows (bands in original matrix)   */
	long nbands_up;   /* Number of bands above diagonal              */
	long nbands_low;  /* Number of bands below diagonal              */
	double *array;    /* Storage for the matrix in banded format     */
	/* Internal temporary storage for solving inverse problem        */
	long nbrows_inv;  /* Number of rows of inverse matrix            */
	double *array_inv;/* Store the matrix decomposition if this is generated:                */
	                  /* this is used to calculate the action of the inverse matrix.         */
	                  /* (what is stored is not the inverse matrix but an equivalent object) */
	int *ipiv;        /* Additional inverse information              */
};
typedef struct band_mat band_mat;

/* Initialise a band matrix of a certain size, allocate memory,
   and set the parameters.  */ 
int init_band_mat(band_mat *bmat, long nbands_lower, long nbands_upper, long n_columns) {
	bmat->nbrows = nbands_lower + nbands_upper + 1;
	bmat->ncol   = n_columns;
	bmat->nbands_up = nbands_upper;
	bmat->nbands_low= nbands_lower;
	bmat->array      = (double *) malloc(sizeof(double)*bmat->nbrows*bmat->ncol);
	bmat->nbrows_inv = bmat->nbands_up*2 + bmat->nbands_low + 1;
	bmat->array_inv  = (double *) malloc(sizeof(double)*(bmat->nbrows+bmat->nbands_low)*bmat->ncol);
	bmat->ipiv       = (int *) malloc(sizeof(int)*bmat->ncol);
	if (bmat->array==NULL||bmat->array_inv==NULL) {
		return 0;
	}
	/* Initialise array to zero */
	long i;
	for (i=0;i<bmat->nbrows*bmat->ncol;i++) {
		bmat->array[i] = 0.0;
	}
	return 1;
};

/* Finalise function: should free memory as required */
void finalise_band_mat(band_mat *bmat) {
	free(bmat->array);
	free(bmat->array_inv);
	free(bmat->ipiv);
}


/* Get a pointer to a location in the band matrix, using
   the row and column indexes of the full matrix.     */
double *getp(band_mat *bmat, long row, long column) {
	int bandno = bmat->nbands_up + row - column;
	if(row<0 || column<0 || row>=bmat->ncol || column>=bmat->ncol ) {
		printf("Indexes out of bounds in getp: %ld %ld %ld \n",row,column,bmat->ncol);
		exit(1);
	}
	return &bmat->array[bmat->nbrows*column + bandno];
}


/* Retrun the value of a location in the band matrix, using
   the row and column indexes of the full matrix.        */
double getv(band_mat *bmat, long row, long column) {
	return *getp(bmat,row,column);
}


double setv(band_mat *bmat, long row, long column, double val) {
	*getp(bmat,row,column) = val;
	return val;
}


/* Solve the equation Ax = b for a matrix a stored in band format
   and x and b real arrays                                          */
int solve_Ax_eq_b(band_mat *bmat, double *h, double *b) {
  /* Copy bmat array into the temporary store */
	int i,bandno;
	for(i=0;i<bmat->ncol;i++) {
		for (bandno=0;bandno<bmat->nbrows;bandno++) {
			bmat->array_inv[bmat->nbrows_inv*i+(bandno+bmat->nbands_low)] = bmat->array[bmat->nbrows*i+bandno];
		}
		h[i] = b[i];
	}

	long nrhs = 1;
	long ldab = bmat->nbands_low*2 + bmat->nbands_up + 1;
	int info = LAPACKE_dgbsv( LAPACK_COL_MAJOR, bmat->ncol, bmat->nbands_low, bmat->nbands_up, nrhs, bmat->array_inv, ldab, bmat->ipiv, h, bmat->ncol);
	return info;
}


// Print the bands of a banded matrix
int print_bmat(band_mat *bmat) {
	long i,j;
	for(i=0; i<bmat->ncol;i++) {
		printf("Col: %ld Row \n",i);
		for(j=0; j<bmat->nbrows; j++) {
			printf("%ld, %g \n",j,bmat->array[bmat->nbrows*i + j]);
		}
	}
	return 0;
}

//reindexing function
long reindex(long index, long ncols) {
	if (index<ncols/2)  {
		return 2*index;
	} else return 2*(ncols-index)-1;  
}

// Diagnostic routines: set to 1 to enable.
#define DIAGS 0

int main(){

	// PARAMETERS:
	// L, right x boundary of domain
	double L;
	// N, number of grid points
	int N;
	// K, wavenumber of forcing
	double K;
	// gamma, damping rate
	double gamma;
	//omega_a, min frequency
	double omega_a;
	//omega_b, max frequency
	double omega_b;
	//I, number of values in freq scan (named z in this code)
	int z;

	// Read in from file; 
	read_inputs(&L, &N, &K, &gamma, &omega_a, &omega_b, &z);

	//dx value
	double dx = L/N;

	//read in E and mu from coefficients.txt and assign them to arrays
	double *E = malloc(sizeof(double)*N);
	double *mu = malloc(sizeof(double)*N);
	long k = 0;
	double u, v;
	FILE *coeffile;
	if ((coeffile = fopen("coefficients.txt","r")) ==NULL){
		printf("Error opening coefficients file");
		exit(1);
	} 
	while (fscanf(coeffile, "%lf %lf", &u, &v) == 2) {
		mu[k] = u;
		E[k] = v;
		k++;
	} 

	fclose(coeffile);
	

	band_mat bmat;
	long ncols = 2*N; //need double N columns to accomodate real and imaginary parts

	/* We have a three-point stencil (domain of numerical dependence) of
	our finite-difference equations:
	1 point to the left  -> nbands_low = 1
	1       to the right -> nbands_up  = 1
	
	after reindexing, the matrix has 4 bands above and below the main band
	*/
	
	long nbands_low = 4;  
	long nbands_up  = 4;	
	init_band_mat(&bmat, nbands_low, nbands_up, ncols);
	double *h = malloc(sizeof(double)*ncols);
	double *b = malloc(sizeof(double)*ncols);
	double *b_reindexed = malloc(sizeof(double)*ncols);

	FILE *outputfile;
	outputfile = fopen("output.txt", "w"); //open output file 


	double omega = omega_a; //omega is the frequency that it outputs at
	while (omega < omega_b){
	
		/* b vector for real (even indexes) and imaginary (odd indexes) parts */

		for (long i=0; i<N; i++){
			b[2*i]    = -cos(K*i*dx); 
			b[2*i +1] = -sin(K*i*dx); 
		}
		
		//reindexing b vector: 
		for(long i=0; i<ncols; i++) {
			b_reindexed[reindex(i,ncols)]=b[i];	
		}


		//setting up matrix
		long i;
		
		E[N] = E[0];  // periodic boundary conditions
		mu[N] = mu[0];

		
		//real equations:	 
		for (i=0; i<N; i++){
			                setv(&bmat, reindex(2*i, ncols), reindex(2*i,    ncols), (-E[i+1]-E[i])/(dx*dx) + omega*omega*mu[i]); 
			if(2*i<ncols-1){setv(&bmat, reindex(2*i, ncols), reindex(2*i +1, ncols), -omega*gamma);};
			if(2*i<ncols-2){setv(&bmat, reindex(2*i, ncols), reindex(2*i +2, ncols), E[i+1]/(dx*dx));}; 
			if(2*i>0){      setv(&bmat, reindex(2*i, ncols), reindex(2*i -2, ncols), E[i]/(dx*dx));}; 
		}
	
		//imaginary equations:
		for (i=0; i<N+1; i++){
			if(2*i>0){      setv(&bmat, reindex(2*i-1, ncols), reindex(2*i -1, ncols), (-E[i]-E[i-1])/(dx*dx) + omega*omega*mu[i-1]);};
			if(2*i>0){      setv(&bmat, reindex(2*i-1, ncols), reindex(2*i -2, ncols), omega*gamma);};
			if(2*i<ncols-2){setv(&bmat, reindex(2*i+1, ncols), reindex(2*i +3, ncols), E[i+1]/(dx*dx));};
			if(2*i>2){      setv(&bmat, reindex(2*i-1, ncols), reindex(2*i -3, ncols), E[i-1]/(dx*dx));};
		}
	 
		//Boundaries conditions (setting top right and bottom left of matrix)
		
		double alpha = (E[0] / (dx*dx)) * cos(K*L);
		double beta  = (E[0] / (dx*dx)) * sin(K*L); 
		
		setv(&bmat, reindex(0, ncols), reindex(ncols-2, ncols), alpha);
		setv(&bmat, reindex(0, ncols), reindex(ncols-1, ncols), beta);
		setv(&bmat, reindex(1, ncols), reindex(ncols-2, ncols), -beta);
		setv(&bmat, reindex(1, ncols), reindex(ncols-1, ncols), alpha);	 
	 
		setv(&bmat, reindex(ncols-2, ncols), reindex(0, ncols), alpha);
		setv(&bmat, reindex(ncols-2, ncols), reindex(1, ncols), -beta);
		setv(&bmat, reindex(ncols-1, ncols), reindex(0, ncols), beta);
		setv(&bmat, reindex(ncols-2, ncols), reindex(1, ncols), alpha);


		/*  Print matrix for debugging: */ 
		if (DIAGS) {
			print_bmat(&bmat);
		}
		
		//if statement to produce NAN's if matrix is singular (solve_Ax_eq_b returns 0 if successful)

		if (solve_Ax_eq_b(&bmat, h, b_reindexed) == 0){
		
			//inverse index to get h in correct order and print

			for(long i=0; i<N; i++) {
				double x = dx*i;	 
	 			fprintf(outputfile,"%lf %lf %lf %lf \n", omega, x, h[reindex(2*i,ncols)], h[reindex(2*i+1, ncols)]);
			}
		} else for(long i=0; i<N; i++) {
				double x = dx*i;	 
	 			fprintf(outputfile,"%lf %lf %lf %lf \n", omega, x, NAN, NAN);
			
		}

		omega += (omega_b-omega_a)/z; //increment omega
	}
 
 
	fclose(outputfile); //close output file
 
	//free storage
	finalise_band_mat(&bmat);
	free(h);
	free(b);
	free(E);
	free(mu);
	free(b_reindexed);
	return 0;
}


//function to read input file
void read_inputs(double *L, int *N, double *K, double *gamma, double *omega_a, double *omega_b, int *z){
	FILE *infile;
	if(!(infile=fopen("input.txt","r"))) {
		printf("Error opening file\n");
		exit(1);
	}
	if(7!=fscanf(infile,"%lf %d %lf %lf %lf %lf %d", L, N, K, gamma, omega_a, omega_b, z)) {
		printf("Error reading parameters from file\n");
		exit(1);
	}
	fclose(infile);
}