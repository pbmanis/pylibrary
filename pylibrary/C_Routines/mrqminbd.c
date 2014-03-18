/* File: mrqminbd.c
   Date: February 22, 1999
   By:   S.C. Molitor

   Implements a Marquardt-Levenberg least-squares minimization to
   find a set of parameters p to fit a nonlinear function f(p, x)
   to a data set.

   Modified from Numerical Recipes in C:
      1) parameters can have upper & lower bounds
      2) the code for invariant parameters is tighter */


// inclusions

#include <math.h>
#include "mex.h"


// definitions

#define	FIRST_ALAMBDA	(0.001)		// first lambda value
#define DEC_ALAMBDA		(0.1)		// lambda decrement
#define INC_ALAMBDA		(10.0)		// lambda increment

#define SORT_FRWD		(1)			// arrange matrix in forward direction
#define SORT_BKWD		(-1)		// arrange matrix in backward direction

#define SWAP(a, b)		{(swap) = (a); (a) = (b); (b) = (swap);}
									// value swap macro


// function prototypes

// ARRANGEFIXED - arrange matrix & vector to account for fixed parameters

void ArrangeFixed(double	**a,	// matrix to arrange
				  double	b[],	// vector to arrange
				  int		npar,	// number of parameters
				  int		ip[],	// indices of fit parameters
				  int		nfit,	// number of fit parameters
				  int		dir);	// direction (-1 or 1)

// GAUSSJ - Find the inverse of a matrix with Gauss-Jordan elimination
//          solves Ax = b, returns inv(A) and solution x
//			returns error value (0 if no error)

int gaussj(double	**a,	// square matrix (replaced by inverse)
		   double	b[],	// rhs vector (replaced by solution x)
		   int		n);		// matrix & vector dimensions

// MRQCOF - calculates curvature matrices, partial derivatives and chi square

void mrqcof(double	x[],		// x values of dataset
			double	y[],		// y values of dataset
			double	sig[],		// individual std. deviations
			int		npts,		// number of data points
			double	p[],		// function parameters p
			int		npar,		// number of function parameters
			int		ip[],		// indices of varying parameters
			int		nfit,		// number of varying parameters
			double	**alpha,	// returned curvature matrix
			double	beta[],		// returned chi^2 partials
			double	*chisq,		// returned chi^2 value
			double	(*funcs)(double,	// function x value, returns f(p, x)
							 double [],	// function parameters p
							 double [],	// returned df/dp values
							 int));		// number of parameters


// MRQMINBD - perform Marquardt-Levenberg least-squares minimization
//            caller must provide f(p, x) in the form y = F(x, p[], dy/dp, np)

void mrqminbd(double	x[],		// x values of dataset
			  double	y[],		// y values of dataset
			  double	sig[],		// individual std. deviations
			  int		npts,		// number of data points
			  double	p[],		// function parameters p
			  double	lb[],		// lower bounds on function parameters
			  double	ub[],		// upper bounds on function parameters
			  int		npar,		// number of function parameters
			  int		ip[],		// indices of parameters to fit
			  int		nfit,		// number of parameters to fit
			  double	**covar,	// returned covariance matrix
			  double	**alpha,	// returned curvature matrix
			  double	beta[],		// returned chi^2 partials matrix
			  double	*chisq,		// returned chi^2 value
			  double	(*funcs)(double,	// function x value, returns f(p, x)
								 double [],	// function parameters p
								 double [],	// returned df/dp values
								 int),		// number of parameters
			  double	*lambda) {	// returned curvature diagonal scaling factor

	// declare variables
	// some are static because of repeated calls

	static double	ochisq;		// old chi^2
	static double	*ptry;		// function parameters to test
	static double	*dp;		// parameter increments
	int				inverr;		// matrix inversion error
	int				j, k;		// loop counters

	// first call has scaling < 0 for initialization

	if (*lambda < 0.0) {
		ptry = (double *) mxCalloc(npar, sizeof(double));
		dp = (double *) mxCalloc(npar, sizeof(double));

		// assign initial parameters

		for (j = 0; j < npar; j++) {
			ptry[j] = p[j];
		}

		// return initial chi^2, curvature & chi^2 partial derivatives

		*lambda = FIRST_ALAMBDA;
		mrqcof(x, y, sig, npts, ptry, npar, ip, nfit, alpha, beta, chisq, funcs);
		ochisq = (*chisq);
		return;
	}

	// augment diagonal curvature values with lambda

	for (j = 0; j < nfit; j++) {
		for (k = 0; k < nfit; k++) {
			covar[j][k] = alpha[j][k];
		}
		covar[j][j] *= 1.0 + (*lambda);
		dp[j] = beta[j];
	}

	// solve for covariance & parameter increments
	// trial is unsuccessful if matrix inversion error

	inverr = gaussj(covar, dp, nfit);
	if (inverr < 0) {
		*lambda *= INC_ALAMBDA;
		return;
	}

	// last call has lambda = 0
	// rearrange matrices & free allocated space

	if (*lambda == 0.0) {
		if (nfit < npar) {
			ArrangeFixed(alpha, beta, npar, ip, nfit, SORT_BKWD);
			ArrangeFixed(covar, dp, npar, ip, nfit, SORT_BKWD);
		}
		mxFree((void *) dp);
		mxFree((void *) ptry);
		return;
	}

	// increment parameters to see if trial succeeded
	// trial is unsuccessful if bounds are exceeded

	for (j = 0; j < nfit; j++) {
		ptry[ip[j]] = p[ip[j]] + dp[j];
		if (mxIsFinite(lb[ip[j]]) && (ptry[ip[j]] <= lb[ip[j]])) {
			*lambda *= INC_ALAMBDA;
			return;
		}
		else if (mxIsFinite(ub[ip[j]]) && (ptry[ip[j]] >= ub[ip[j]])) {
			*lambda *= INC_ALAMBDA;
			return;
		}
	}

	// obtain updated chi^2, curvature & chi^2 partial derivatives
	// note that covar & dp are used to preserve alpha & beta if step fails

	mrqcof(x, y, sig, npts, ptry, npar, ip, nfit, covar, dp, chisq, funcs);

	// if successful, accept new parameters
	// decrease lambda

	if (*chisq < ochisq) {
		*lambda *= DEC_ALAMBDA;
		ochisq = (*chisq);
		for (j = 0; j < nfit; j++) {
			for (k = 0; k < nfit; k++) {
				alpha[j][k] = covar[j][k];
			}
			beta[j] = dp[j];
			p[ip[j]] = ptry[ip[j]];
		}
	}

	// otherwise reject new parameters
	// increase lambda

	else {
		*lambda *= INC_ALAMBDA;
		*chisq = ochisq;
	}
}


// ARRANGEFIXED - arrange matrix & vector to account for fixed parameters

void ArrangeFixed(double	**a,	// matrix to arrange
				  double	b[],	// vector to arrange
				  int		npar,	// number of parameters
				  int		ip[],	// indices of fit parameters
				  int		nfit,	// number of fit parameters
				  int		dir) {	// direction (SORT_BKWD or SORT_FRWD)

	// declare variables

	double	swap;		// storage for swapping values
	int		i, j, k;	// loop counters

	// zero values for fixed parameters if backward direction

	if (dir == SORT_BKWD) {
		for (i = nfit; i < npar; i++) {
			for (j = 0; j <= i; j++) {
				a[i][j] = 0.0;
				a[j][i] = 0.0;
			}
			b[i] = 0.0;
		}
	}

	// relocate values into the appropriate positions
	// account for forward/backward direction

	for (i = 0; i < nfit; i++) {

		// compute index for forward/backward direction

		k = (dir == SORT_BKWD) ? nfit - 1 - i : i;

		if (ip[k] != k) {

			// swap rows

			for (j = 0; j < npar; j++) {
				SWAP(a[k][j], a[ip[k]][j]);
			}

			// swap columns

			for (j = 0; j < npar; j++) {
				SWAP(a[j][k], a[j][ip[k]]);
			}

			// swap vector values

			SWAP(b[k], b[ip[k]]);
		}
	}

	// zero values for fixed parameters if forward direction

	if (dir == SORT_FRWD) {
		for (i = nfit; i < npar; i++) {
			for (j = 0; j <= i; j++) {
				a[i][j] = 0.0;
				a[j][i] = 0.0;
			}
			b[i] = 0.0;
		}
	}
}


// GAUSSJ - Find the inverse of a matrix with Gauss-Jordan elimination
//          solves Ax = b, returns inv(A) and solution x
//			returns error value (0 if no error)

int gaussj(double	**a,	// square matrix (replaced by inverse)
		   double	b[],	// rhs vector (replaced by solution x)
		   int		n) {	// matrix & vector dimensions

	// declare variables

	double	maxval;			// max value in current row
	double	invpivot;		// inverse of pivot element
	double	reduceval;		// value for reducing non-pivot rows
	double	swap;			// swap storage
	int		*indexc;		// column index for pivot element
	int		*indexr;		// row index for pivot element
	int		*ipivot;		// flag to indicate pivot column
	int		icol, irow;		// location of pivot element
	int		i, j, k;		// loop counters

	// allocate storage for index arrays
	// used in pivoting operation

	indexc = mxCalloc(n, sizeof(int));
	indexr = mxCalloc(n, sizeof(int));
	ipivot = mxCalloc(n, sizeof(int));

	// main loop for columns to be reduced

	for (i = 0; i < n; i++) {
		maxval = 0.0;

		// find pivot element (= MAX(|a(j,k)|))
		// insure same element not used more than once

		for (j = 0; j < n; j++) {
			if (ipivot[j] != 1) {
				for (k = 0; k < n; k++) {
					if (ipivot[k] == 0) {
						if (fabs(a[j][k]) >= maxval) {
							maxval = fabs(a[j][k]);
							irow = j;
							icol = k;
						}
					}
					else if (ipivot[k] > 1) {
						return(-1);
					}
				}
			}
		}
		ipivot[icol]++;

		// interchange rows to put pivot on diagonal if needed
		// index arrays store where interchanges occurred

		if (irow != icol) {
			for (k = 0; k < n; k++) {
				SWAP(a[irow][k], a[icol][k]);
			}
			SWAP(b[irow], b[icol]);
		}
		indexr[i] = irow;
		indexc[i] = icol;

		// divide pivot row by pivot element

		if (a[icol][icol] == 0.0) {
			return(-2);
		}
		invpivot = 1.0 / a[icol][icol];
		a[icol][icol] = 1.0;
		for (k = 0; k < n; k++) {
			a[icol][k] *= invpivot;
		}
		b[icol] *= invpivot;

		// reduce non-pivot rows

		for (j = 0; j < n; j++) {
			if (j != icol) {
				reduceval = a[j][icol];
				a[j][icol] = 0.0;
				for (k = 0; k < n; k++) {
					a[j][k] -= a[icol][k] * reduceval;
				}
				b[j] -= b[icol] * reduceval;
			}
		}
	}

	// unscramble columns in reverse order of which
	// they were originally scrambled!

	for (k = (n - 1); k >= 0; k--) {
		if (indexr[k] != indexc[k]) {
			for (j = 0; j < n; j++) {
				SWAP(a[j][indexr[k]], a[j][indexc[k]]);
			}
		}
	}

	// free allocated arrays

	mxFree((void *) ipivot);
	mxFree((void *) indexr);
	mxFree((void *) indexc);
	return(0);
}


// MRQCOF - calculates curvature matrices, partial derivatives and chi square

void mrqcof(double	x[],		// x values of dataset
			double	y[],		// y values of dataset
			double	sig[],		// individual std. deviations
			int		npts,		// number of data points
			double	p[],		// function parameters p
			int		npar,		// number of function parameters
			int		ip[],		// indices of varying parameters
			int		nfit,		// number of varying parameters
			double	**alpha,	// returned curvature matrix
			double	beta[],		// returned chi^2 partials
			double	*chisq,		// returned chi^2 value
			double	(*funcs)(double,	// function x value, returns f(p, x)
							 double [],	// function parameters p
							 double [],	// returned df/dp values
							 int)) {	// number of parameters

	// declare variables

	double	*dfdp;		// partial derivatives df/dp
	double	yf;			// evaluated y = f(p, x)
	double	sig2i;		// inverse of sig^2
	double	dy;			// y(x) - f(p, x)
	double	wt;			// scaling for curvature values
	int		i, j, k;	// loop counters

	// allocate space for partial derivatives

	dfdp = (double *) mxCalloc(npar, sizeof(double));

	// initialize lower curvature matrix
	// curvature matrix is symmetric by definition
	// initialize vector for chi^2 partial derivatives

	for (j = 0; j < nfit; j++) {
		for (k = 0; k <= j; k++) {
			alpha[j][k] = 0.0;
		}
		beta[j] = 0.0;
	}

	// loop through data to calculate chi^2

	*chisq = 0.0;
	for (i = 0; i < npts; i++) {
		yf = (*funcs)(x[i], p, dfdp, npar);
		sig2i = 1.0 / (sig[i] * sig[i]);
		dy = y[i] - yf;
		*chisq += dy * dy * sig2i;

		// calculate curvature & chi^2 partials
		// takes advantage of symmetric curvature matrix

		for (j = 0; j < nfit; j++) {
			wt = dfdp[ip[j]] * sig2i;
			for (k = 0; k <= j; k++) {
				alpha[j][k] += wt * dfdp[ip[k]];
			}
			beta[j] += dy * wt;
		}
	}

	// normalize chi^2 to degrees of freedom (technically incorrect)
	// provides direct assessment of chi^2 ~= D.O.F. (chi^2 / D.O.F. ~= 1)

	if (npts > nfit) {
		*chisq /= (double) (npts - nfit);
	}

	// assign remainder of curvature matrix values

	for (j = 1; j < nfit; j++) {
		for (k = 0; k < j; k++) {
			alpha[k][j] = alpha[j][k];
		}
	}

	// free allocated space

	mxFree((void *) dfdp);
}
