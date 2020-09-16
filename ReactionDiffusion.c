#include <stdlib.h>
#include <stdio.h>
#include <lapacke.h>
#include <math.h>

void read_inputs(long *Nx, long *Ny, double *Lx, double *Ly, double *tf, double *lambda, double *sigma, double *kappa, double *A0, double *A1, double *A2, double *t_D);

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

/*Check that a grid point has valid coordinates */
int is_valid(long j, long p, long J, long P) {
  return (j>=0)&&(j<J)&&(p>=0)&&(p<P);
}


/* Return the 1D element index corresponding to a particular grid point.
	We can rewrite this function without changing the rest of the code if
	we want to change the grid numbering scheme!
	Output: long integer with the index of the point
	Input:
	long x:  The X grid point index
	long y:  The Y grid point index
	long X:  The number of x points.
	long Y: The number of y points.
*/


//to get smallest banded matrix possible, if function is used in the indexing
long indx(long x, long y, long X, long Y) {

	if (x == -1){
		x = X-1;
	}
	if (x == X){
		x = 0;
	}
	if (y == -1){
		y = 1;
	}
	if (y == Y){
		y = Y-2;
	} 
	
	if (X<Y){
		return x + X*y;
	} 
	else { return y + Y*x;
	}
}

/* Return the 2D point corresponding to a particular 1D grid index */
void gridp(long indx, long P, long *j, long *p) {
	*j = indx%P;
	*p = indx - (*j)*P;
}


// Function to neatly print the full matrix. Used primarily in the
// debugging/verifying stages.
int print_mat(band_mat *bmat){
	long row,column;
	printf("       ");
	for(column=0; column<bmat->ncol; column++){
		printf("%11ld ",column);
	}
	printf("\n");
	printf("       ");
	for(column=0; column<bmat->ncol; column++){
		printf(" ---------- ");
	}
	printf("\n");

	for(row=0; row<bmat->ncol;row++){
		printf("%4ld : ",row);
		for(column=0; column<bmat->ncol; column++){
			double flval = 0.0;
			int bandno = bmat->nbands_up + row - column;
			if( bandno<0 || bandno>=bmat->nbrows )
				{}
			else{
				flval = getv(bmat,row,column);
			}
			printf("%11.4g ",flval);
		}
		printf("\n");
	}
	return 0;
}

// Diagnostic routines: set to 1 to enable.
#define DIAGS 0

int main(){

	// PARAMETERS:
	//defining pi
	#define M_PI acos(-1.0) 
	// Number of x and y gridpoints
	long Nx, Ny;
	// Length of x and y domains
	double Lx, Ly;
	//final time
	double tf;
	//parameters in equations
	double lambda, sigma, kappa;
	//intialisation parameters
	double A0, A1, A2;
	//diagnostic timestep
	double t_D;

	// Read in from file; 
	read_inputs(&Nx, &Ny, &Lx, &Ly, &tf, &lambda, &sigma, &kappa, &A0, &A1, &A2, &t_D);

	//dx and dy values
	double dx = Lx/Nx;
	double dy = Ly/(Ny-1);


	band_mat bmat;
	long ncols = Nx*Ny; 
	
	
	//the matrix has different number of bands depending on what indx function is used
	long nbands_low, nbands_up;
	if (Nx<Ny){
		nbands_low = Nx;
		nbands_up  = Nx;
	} 
	else { nbands_low = Ny;
		 nbands_up  = Ny;
		}
		

	init_band_mat(&bmat, nbands_low, nbands_up, ncols);
	double *u = malloc(sizeof(double)*(ncols));
	double *v = malloc(sizeof(double)*(ncols));
	double *u_next = malloc(sizeof(double)*(ncols));
	double *v_next = malloc(sizeof(double)*(ncols));
	double *bu = malloc(sizeof(double)*ncols);
	double *bv = malloc(sizeof(double)*ncols);
	
	FILE *outputfile;
	outputfile = fopen("output.txt", "w"); //open output file 	
	
	//initialising u and v
	
	for (long i=0; i<Ny; i++){
		double y = dy*i;
			for (long l=0; l<Nx; l++){
			double x = dx*l;
			u[indx(l, i, Nx, Ny)] = (A0 + A1*cos(M_PI*y/Ly))*exp(-pow((A2*x/Lx), 2));
			v[indx(l, i, Nx, Ny)] = 0;
		}
	}
 
	double dt; //timestep
	double ctime = 0.0;
	double next_output_time = t_D;


	
	//print at t=0
	for(long i=0; i<Ny; i++) {
	double y = dy*i;
		for(long l=0; l<Nx; l++){
			double x = dx*l;	 
			fprintf(outputfile,"%lf %lf %lf %lf \n", x, y, u[indx(l, i, Nx, Ny)], v[indx(l, i, Nx, Ny)]);
		}
	}

	
	//START OF WHILE LOOP
	
	while (ctime < tf){
		
		//stable timestep
		if(dx>1 || dy>1){
			dt = 0.01; 
		} 
		else {
			dt = 0.1;
		}
		
		
		// If we would go past the next output step, reduce the timestep.
		double dt_new = dt;
		int output = 0;
		
		if (ctime+dt_new>next_output_time) {
			dt_new = next_output_time - ctime;
			output = 1;
		}
	
		
		//setting up matrix		
		for (long i=0; i<Ny; i++){
			for (long j = 0; j<Nx; j++){ 
				
				setv(&bmat, indx(j,i,Nx,Ny), indx(j,i,Nx,Ny), (2/(dt_new) + 2/(dx*dx) + 2/(dy*dy))); 
				setv(&bmat, indx(j,i,Nx,Ny), indx(j-1,i,Nx,Ny), -1/(dx*dx));
				setv(&bmat, indx(j,i,Nx,Ny), indx(j+1,i,Nx,Ny), -1/(dx*dx)); 
				setv(&bmat, indx(j,i,Nx,Ny), indx(j,i-1,Nx,Ny), -1/(dy*dy)); 
				setv(&bmat, indx(j,i,Nx,Ny), indx(j,i+1,Nx,Ny), -1/(dy*dy)); 
				
				
				//x boundaries overlap in for Nx=2 matrix, setv 2*x
				if (Nx == 2){
					if ((j-1) == (j+1)){
						setv(&bmat, indx(j,i,Nx,Ny), indx(j+1,i,Nx,Ny), -2/(dx*dx));
					}
				}
				
				//y boundaries overlap with main diagonal for Ny=2
				if(Ny == 2){
					if ((i+1) == Ny){
						setv(&bmat, indx(j,i,Nx,Ny), indx(j,i,Nx,Ny), (2/(dt_new) + 2/(dx*dx) + 2/(dy*dy)) -1/(dy*dy));
					}
				}
				
				//boundary conditions for y to account for overlaps
				if ((i-1) == -1){ 
					setv(&bmat, indx(j,i,Nx,Ny), indx(j,1,Nx,Ny), -2/(dy*dy));
				}
				
				if ((i+1) == Ny){
					setv(&bmat, indx(j,i,Nx,Ny), indx(j,Ny-2,Nx,Ny), -2/(dy*dy));
				} 
			}
		}


		/*  Print matrix for debugging: */ 
		if (DIAGS) {
			print_bmat(&bmat);
		}
		
		/* b vectors */
		for (long i=0; i<Ny; i++){
			for (long j= 0; j<Nx; j++){
				bu[indx(j, i, Nx, Ny)] = 2*(u[indx(j, i, Nx, Ny)]*(lambda + 1/dt_new) - pow(u[indx(j, i, Nx, Ny)], 3) - kappa - sigma*v[indx(j, i, Nx, Ny)])  + u[indx(j-1, i, Nx, Ny)]/(dx*dx) + u[indx(j+1,i, Nx, Ny)]/(dx*dx) + u[indx(j, i-1, Nx, Ny)]/(dy*dy) + u[indx(j, i+1, Nx, Ny)]/(dy*dy) - u[indx(j, i, Nx, Ny)]*(2/(dx*dx) + 2/(dy*dy));
				bv[indx(j, i, Nx, Ny)] = 2*(v[indx(j, i, Nx, Ny)]*(-1     + 1/dt_new) + u[indx(j, i, Nx, Ny)]) + v[indx(j-1, i, Nx, Ny)]/(dx*dx) + v[indx(j+1,i, Nx, Ny)]/(dx*dx) + v[indx(j, i-1, Nx, Ny)]/(dy*dy) + v[indx(j, i+1, Nx, Ny)]/(dy*dy) - v[indx(j, i, Nx, Ny)]*(2/(dx*dx) + 2/(dy*dy));
			}
		}
		
		
		//solve to get next timestep values

		solve_Ax_eq_b(&bmat, u_next, bu);
		solve_Ax_eq_b(&bmat, v_next, bv);

		// copy next timestep values to u and v to use for next iteration
		for(long i=0; i<Nx; i++){
			for(long j=0; j<Ny; j++){
				u[indx(i,j, Nx, Ny)] = u_next[indx(i,j,Nx,Ny)];
				v[indx(i,j, Nx, Ny)] = v_next[indx(i,j,Nx,Ny)];
			}
		}
			
		
		ctime += dt_new;
		//increment time and print at td
		if (output) {
			for(long i=0; i<Ny; i++) {
				double y = dy*i;
				for(long l=0; l<Nx; l++){
					double x = dx*l;
					fprintf(outputfile,"%lf %lf %lf %lf \n", x, y, u[indx(l, i, Nx, Ny)], v[indx(l, i, Nx, Ny)]);
				}
			}
			next_output_time += t_D;
		}

	}  //END OF WHILE LOOP
		
 
 
	fclose(outputfile); //close output file
 
	//free storage
	finalise_band_mat(&bmat);
	free(u);
	free(v);
	free(u_next);
	free(v_next);
	free(bu);
	free(bv);
	return 0;
}


//function to read input file
void read_inputs(long *Nx, long *Ny, double *Lx, double *Ly, double *tf, double *lambda, double *sigma, double *kappa, double *A0, double *A1, double *A2, double *t_D){
	FILE *infile;
	if(!(infile=fopen("input.txt","r"))) {
		printf("Error opening file\n");
		exit(1);
	}
	if(12!=fscanf(infile,"%ld %ld %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", Nx, Ny, Lx, Ly, tf, lambda, sigma, kappa, A0, A1, A2, t_D)) {
		printf("Error reading parameters from file\n");
		exit(1);
	}
	fclose(infile);
}