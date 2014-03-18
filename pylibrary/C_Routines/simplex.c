/*	------------------------------------------------------------


		Modified version of simp from byte march 1984
	for the DATAC program.
		The parameters are passed into var[] and the mean result
	is returned into it.

		The theoretical equation and the data are taken from a strip

	Update 18-11-1985	D.B. replaced pow( ,2.) by *
	Important Update for the data entry and display
	September 1986 D.Bertrand

	Update 11/22/88 - Let blanking on strip be used to delete points
		from the lms computation in the fit (P. Manis )

	Modified to work with getcmd.c and MSC V5.0 P. Manis, 1/90.

	Update 24-Apr-90 Added weighting function to lms fit -  you specify strip.

	-------------------------------------------------------------
*/
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <mpak87.h>
#include <malloc.h>

#include "strip.h"
#include "datach.h"

extern float * load_mask_special(char *s, int *flag); /* in stripmgr.c */

extern set_theoric (char *s, int *nvar, DSTRIP *xstrip);

static void sum_res(double x[]);
static void order(void);
static void first(void);
static void new_vertex(void);
static void report(void);
static int simplex_fit(void);
static int marq_fit(void);

#define FOREVER for(;;)
#define M 9		/* maximum number of parameters to fit */
#define NVP 2	/*total number of variables per data pts */
#define ALFA 1. /* reflection coeficient > 0 */
#define BETA .5 /* contraction coeficient 0 to 1 */
#define GAMMA 2./* expansion coeficient >1 */
#define ROOT2 1.414214
#define N M+1
#define LOOP for(i = 0; i < n ; i++)

#define SIMPLEX 0
#define MARQUARDT 1

int mrq_eqn = 1;		/* set the marquardt equation */

extern char recalc_flag;/* interdependent computation */
extern DSTRIP *droot;
extern double var[MAXVAR];
extern int demo;
extern int macro_level;
extern char temp[MAX_LINE];	/* temporary character strip */

static float * vd = NULL;
static float * vr = NULL;
static float * vb = NULL;	/* the blanking array with special mode (0 to mask, 1 to leave intact) */
static float * vx = NULL;
static int npts = 0;
static int mask_flag = 0;	/* mask flag */
static int done=0;/* end iteration flag */
static ndisp=100;/* number for the display */
static int nrnd=0;
static int mode = SIMPLEX;
static int ssfirst = 0;	/* 0 until run once */

static int skipcnt = 1;	/* skip counter: calcuate rms at skipped points */
static int ncount = 1;
static int np = 0,maxiter = 0,niter = 10;
static double next[N],center[N],mean[N],error[N],maxerr[N];
static double p[N],q[N],step[N];
static double simp[N][N];/* simplex array */
static int h[N],l[N];
double lb[N], hb[N];	/* lower and upper bounds for values in fit */
static DSTRIP *dx = NULL;
static DSTRIP *dy = NULL;
static DSTRIP *py = NULL;
static DSTRIP *wx = NULL;/* for x and y data and theoretical strip, and weighting */
static char xname[LABEL],yname[LABEL],theoric[LABEL],weighting[LABEL];
static int n,m;

struct command simparam[]={	/* simplex commands */
	"exit",1,0,
	"xdata",2,0,
	"ydata",3,0,
	"yth",4,0,
	"niter",5,0,
	"ndisp",6,0,
	"rnd",7, 0,
	"nvar",8,0,
	"maxerrs",9,0,
	"run",10, 0,
	"v0",11,  0,
	"v1",12,  0,
	"v2",13,  0,
	"v3",14,  0,
	"v4",15,  0,
	"v5",16,  0,
	"v6",17,  0,
	"v7",18,  0,
	"i0",19,  0,
	"i1",20,  0,
	"i2",21,  0,
	"i3",22,  0,
	"i4",23,  0,
	"i5",24,  0,
	"i6",25,  0,
	"i7",26,  0,
	"e0",27,  0,		/* error for given variable */
	"e1",28,  0,
	"e2",29,  0,
	"e3",30,  0,
	"e4",31,  0,
	"e5",32,  0,
	"e6",33,  0,
	"e7",34,  0,
	"l0",37,  0,	/* low bound for variable */
	"l1",38,  0,
	"l2",39,  0,
	"l3",40,  0,
	"l4",41,  0,
	"l5",42,  0,
	"l6",43,  0,
	"l7",44,  0,
	"h0",45,  0,	/* high bound for variable */
	"h1",46,  0,
	"h2",47,  0,
	"h3",48,  0,
	"h4",49,  0,
	"h5",50,  0,
	"h6",51,  0,
	"h7",52,  0,
	"weighting",53,0,	/* specify weighting function */
	"skipcnt",54,0,	/* skip counter  (usually 1; sometimes bigger for faster cycle) */
	"simplex",55,0,	/* set mode to simplex */
	"marquardt",56,0,
	"equation",57,0, /* built-in equation selection for marquardt (includes predefined derivatives) */
	"continue",35,0,	/* continue the fit */
	"pause",36,0,	/* program pause for the demo */
	"help",99,0,	/* list current command list */
	"?",99,1,
};
#define SPS (sizeof(simparam)/ sizeof (struct command))



struct command mrqeqn[] = {
	"exp1",1,0,  // one exponential
	"exp2",2,0,  // two exponentials
	"exp3",3,0,  // three exponentials
	"expd1",4,0,  // exp with delay built in
	"expd2",5,0,
	"expd3",6,0,
	"boltz",7,0,  // boltzmann eqn.
	"help",99,0,
};

#define SEQ (sizeof(mrqeqn)/sizeof (struct command))


/*	compute summ of square of diff. between theoretical and data */

static void sum_res(double x[])
{
#define MASK 1	/* blanking mask */
#define SELECT 0x80 /* select mask */

	float *vt;
	int i;

	vt = NULL;
	i = 0;

	/* install limits on x[i] for acceptable values for computation, if necessary */
	for(i=0 ; i < m ; i++) {
		if(x[i] < lb[i]) x[i] = lb[i];	/* also modify values for computation */
		if(x[i] > hb[i]) x[i] = hb[i];
		var[i] = x[i];/* transfer the parameters for the equation */
	}

	command(py->func,2);/* compute the equation - run silently unless there's an error! */

	x[m]=0.;/* rezero the sum */

//	load linear arrays for fast computation of error
	vt = load_array(py->label); 	/* theoretical result from the fit */
	vsubv_s(vr, vt, vd, npts);	/* theor - dat */
	sqrv_s(vt, vr, npts);		/* (theor - dat) ^ 2 */
	if(mask_flag > 0) vxv_s(vt, vt, vb, npts);		// zero the masked points so we don't use them in the
													// determination of the fit
	x[m] = sumv_s(vt, npts);	/* sum of squared difference is total lms error */
	_ffree(vt);					// release the temporary array
	return;
}

#define ORIGDB 1
static void order(void)
{/*	gives high/low in each parameter in simp
	caution: simp data not initialized
*/
	int i,j;
	i = j = 0;
#ifdef ORIGDB
// original DB code:
	for(j=0 ; j < n ; j++) { /* for all vertex */
		LOOP { /* of all vertex find the best and the worst */
			if(simp[i][j] < simp[l[j]][j]) l[j]=i;
			if(simp[i][j] > simp[h[j]][j]) h[j]=i;
		}
	}
#endif
#ifndef ORIGDB

// try this:

		LOOP { /* of all vertex find the best and the worst */
			if(simp[i][m] < simp[l[m]][m]) l[m]=i;
			if(simp[i][m] > simp[h[m]][m]) h[m]=i;
		}
#endif

}

static void first(void)
{
	int i,j;
	i = j = 0;
	mvprint(1,ERRLIN+1,"Starting simplex");
	ncount = 1;
	for(j=0 ; j < n ; j++) {
		mvprint(1,-1,"Simp [%2d]  ",j);
		LOOP
			mvprint(0,0,"%9.3g  ",simp[j][i]);
	}
}

static void new_vertex(void)
{/* next in place of the worst vertex	*/
	int i;
	i = 0;
	if(!(niter % ndisp))	{
		if(niter > 0) erase_dial();
	   mvprint(1,ERRLIN+1,"-- Simplex Iteration %5d --",niter);
	}
	for(i = 0; i < n; i++) {
/*
	if(i < n-1) {
			if(next[i] < lb[i]) next[i] = lb[i];
			if(next[i] > hb[i]) next[i] = hb[i];
		}
*/
		simp[h[m]][i] = next[i];
		if(!(niter % ndisp))
			if(i < (n-1)) mvprint(1,-1,"n[%1d]    %8.3g    %8.3e",i,next[i],error[i]);
			else mvprint(1,-1,"lms:    %8.3g",next[n-1]); /* for that last one */
	}
}

static double maxer = 0.0;

int enter()
{/*	-----------------------------------------------------------
	Enter the parameters and display all
	the information in the usual strip display mode
	------------------------------------------------------------*/

	int i,j,k;
    float chisq;

	graf_off();

	i = j = k = 0;
	chisq = (float) var[m];
	FOREVER { /*	display if needed and wait the entry	*/
	if(eolcmd() && !macro_level && (j != 99)) { /* display the informations 	*/
		erase_dial();
		if(mode == SIMPLEX) mvprint(-1,HELPLIN+1,"SIMPLEX Mode");
		if(mode == MARQUARDT) mvprint(-1,HELPLIN+1,"MRQMIN Mode");
		mvprint(1,ERRLIN+1,"  xdata%10s   ydata%10s   weighting%10s",xname,yname,weighting);
		mvprint(1,-1,"yth  %10s  ---- function  ----",theoric);
		if(py) mvprint(1,-1,"%s",py->func);
		mvprint(1,-1,"  niter%7d  ndisp%5d  rnd%5d  skip%5d",maxiter,ndisp,nrnd,skipcnt);
		mvprint(1,-1,"  nvar%5d ",m);/* number of variables */
		mvprint(1,-1,"var# init Value   Increment   Err    LowBound   HiBound     n");
		for(i=0 ; i < m ; i++)
			/*	display all variables		*/
			mvprint(1,-1,"%1d %9.3g  %9.3g  %9.3g  %9.3g  %9.3g  %9.3g",i,simp[0][i],step[i],maxerr[i],lb[i],hb[i],var[i]);
		mvprint(1,-2,"maxerrs %.3g     last chisq: %.3g",maxerr[m], chisq);
		mvprint(1,-2,"Type CONTINUE to fit with stored variables");
		}
	if(cstat()) break;/* abort command with CTRL-C */

	if(mode == SIMPLEX) pcmd = nxtcmd(1,CMDLIN,"Simplex > ");
	if(mode == MARQUARDT) pcmd = nxtcmd(1,CMDLIN,"Marquardt> ");

	switch(j = cmpcmds(pcmd,simparam,SPS)) { /* interpret */
	case 1:
		forcecmd();/* no more information pending */
		erase_dial();
		graf_on();
		return(-1);/* return in the main program */
	case 2:
		sentry(xname,LABEL,1,CMDLIN,"X strip name: ");/* get the xname strip */
		if((dx=dscan(droot,xname))==NULL) strcpy(xname," ");
		break;
	case 3:
		sentry(yname,LABEL,1,CMDLIN,"Y strip name: ");
		if((dy=dscan(droot,yname))==NULL) strcpy(yname," ");
		break;
	case 4:
		if(mode == MARQUARDT) {
			mvprint(1,ERRLIN,"Set theoretical eqn with the equation command");
			forcecmd();
			break;
		}
		sentry(theoric,LABEL,1,CMDLIN,"Theoretical strip: ");
		if((py=dscan(droot,theoric))==NULL) strcpy(theoric," ");
		break;
	case 5:
		maxiter=ientry(maxiter,1,CMDLIN,"Iterations: ");/* nb iterations */
		maxiter=(maxiter<=0 || maxiter>8000) ? 200 :maxiter;
		break;
	case 6:
		ndisp=ientry(ndisp,1,CMDLIN,"Iterations per update: ");
		ndisp=(ndisp <=0 || ndisp >500) ? 100 : ndisp;
		break;
	case 7:
		nrnd=ientry(nrnd,1,CMDLIN,"Nrnd: ");
		break;
	case 8:
		m=ientry(m,1,CMDLIN,"# variables: ");/* number of variables */
		m=(m <= 0 || m > 9) ? 8 : m;
		break;
	case 9:	/* set max err on final */
		maxerr[m]=(double)fentry((float)maxerr[m],1,CMDLIN,"Max error: ");/* maximum error */
		maxer=maxerr[m];
		break;
	case 35:/*continue the fit reload the init value*/
		erase_dial();	/* first clear the screen */
		for(j=0; j < m; j++)
			simp[0][j]=var[j];

	case 10:	/*	run after setting the data	*/
		maxiter = (maxiter == 0) ? 100 : maxiter;
		if(m == 0 ) return (-1);
		n=m+1;/* number of variables +1 */
		maxerr[m]=maxer;
		if((dy=dscan(droot,yname))==NULL ) return(-1);
		if((dx=dscan(droot,xname))==NULL ) return(-1);
		if((py=dscan(droot,theoric))==NULL ) return(-1);
		np=size_strip_pt(dy);
		return(0);/* ok Go */
	case 11:	/* initial variables entry */
	case 12:
	case 13:
	case 14:
	case 15:
	case 16:
	case 17:
	case 18:
		k = j-11;
		sprintf(temp,"Starting value for v%d: ",k);
		simp[0][k]=(double)fentry((float)simp[0][k],1,CMDLIN,temp);
		break;
	case 19:	/* step size */
	case 20:
	case 21:
	case 22:
	case 23:
	case 24:
	case 25:
	case 26:
		k = j-19;
		sprintf(temp,"Step for v%d: ",k);
		step[k]=(double)fentry((float)step[k],1,CMDLIN,temp);
		break;
	case 27:	/* error */
	case 28:
	case 29:
	case 30:
	case 31:
	case 32:
	case 33:
	case 34:
		k = j-27;
		sprintf(temp,"Maxerr for v%d: ",k);
		maxerr[k]=(double)fentry((float)maxerr[k],1,CMDLIN,temp);
		break;
	case 37:	/* lower boundary entry */
	case 38:
	case 39:
	case 40:
	case 41:
	case 42:
	case 43:
	case 44:
		k = j-37;
		sprintf(temp,"Min value for V%d: ");
		lb[k]=(double)fentry((float)lb[k],1,CMDLIN,temp);
		break;
	case 45:	/* lower boundary entry */
	case 46:
	case 47:
	case 48:
	case 49:
	case 50:
	case 51:
	case 52:
		k = j-45;
		sprintf(temp,"Max value for v%d: ",k);
		hb[k]=(double)fentry((float)hb[k],1,CMDLIN,temp);
		break;
	case 53:
		sentry(weighting,LABEL,1,CMDLIN,"Weighting strip: ");
		if((wx=dscan(droot,weighting))==NULL) strcpy(weighting," ");
		break;
	case 36:
		pause();/* program pause for the demo */
		break;
	case 54:
		skipcnt=ientry(skipcnt,1,CMDLIN,"Skip pts: ");/* skip number of points */
		skipcnt=(skipcnt<=0 || skipcnt>256) ? 1 : skipcnt;
		break;
	case 55:
		mode = SIMPLEX;
		break;
	case 56:
		mode = MARQUARDT;
		break;
	case 57:   // select the mrq equation to use
		if(dx == NULL) {
			mvprint(1,ERRLIN, "Must have x selected before setting equation");
			forcecmd();
			break;
		}pcmd = nxtcmd(1,CMDLIN,"Equation sel: ");
		k = cmpcmds(pcmd,mrqeqn,SEQ);
		if(k >= 1 && k <=3 ) {  // limit equations here
			mrq_eqn = k;
			set_theoric(theoric, &m, dx);	// store the equation in theoric and return the number of variables
			py=dscan(droot,theoric);   // get a pointer to the function
			break;
		}
		if(k == 99) {
			erase_dial();
			graf_off();
			listcmds(2,mrqeqn,SEQ);
			break;
		}
		break;

	case 99:
		erase_dial();	/* clear the screen we're going to write on */
		graf_off();
		listcmds(2,simparam, SPS);
		break;
	default:
		if(!check(pcmd)) return(-1);
		empty();
		break;
	}
   }
	return(0);/* abort command*/
}

static void report(void)
{/* reports the information	*/
	int i,j;
	double sigma = 0.0;

	i = j = 0;
	erase_dial();
	mvprint(1,ERRLIN+1,"End of simplex iterations");
	mvprint(1,-2,"Total iterations: %d",niter);
	mvprint(1,-1,"Summary of results:");
	for(j = 0 ; j < n ; j++) {
		mvprint(1,-1,"Simp [%2d]  ",j);
		LOOP
			mvprint(0,0,"%7.3g  ",simp[j][i]);
	}
	mvprint(1,-2,"The final mean values & errors are: ");
	for(i=0; i < n-1; i++)
		mvprint(1,-1,"n[%1d] %9.3g     frac err: %9.3g",i,mean[i],error[i]);/* final values */
	sum_res(mean);/* compute it with final parameters */
	sigma=mean[m];
	sigma = sqrt((sigma/(double)np));
	mvprint(1,-1,"The standard deviation is:  %9.3g",sigma);
	if(np >  m) sigma /= sqrt((double) (np - m));
		else sigma /= sqrt((double)(m-np));
	mvprint(1,-1,"The estimated error is:     %9.3g",sigma);
	var[m] = mean[m];	/* save the result here also */
	var_save(droot);		/* also store new variables */
}

static char save_flag;

void curve_fit(void)
{
	int i,j;

	save_flag=recalc_flag;/* save the actual flag */
	i = j = 0;
	recalc_flag=0;/* clear the recalc flag during simplex */
	if(!ssfirst) {		/* make sure variables are initialized */
		for(i = 0; i < N; i++) {
			lb[i] = -(FLT_MAX)/2.0;
			hb[i] = (FLT_MAX)/2.0;
			next[i] = center[i] = mean[i] = error[i] = maxerr[i] = 0.0;
			p[i] = q[i] = step[i] = 0.0;
			for(j = 0; j < N; j++) simp[i][j] = 0.0;
		}
		ssfirst = 1;
	}
	if(enter() != 0) { /* bad entry return to the main */
		recalc_flag=save_flag;
		return;
	}
	if(mode == SIMPLEX) simplex_fit();
	if(mode == MARQUARDT) marq_fit();
}

static 	float chisq, alambda, a[20], dyda[20], covar[20][20], alpha1[20][20];

extern void mrq_fnc();

static int marq_fit(void)
{

	int i,j, lista[20];
	float ochisq = (float) 0.0;

	npts = size_strip_pt(dy);
	if(npts != size_strip_pt(dx)) {
		mvprint(1,ERRLIN,"Strips do not have matched points");
		return(-1);
	}
	vx = load_array(dx->label);
	vd = load_array(dy->label); /* raw data */
	vb = load_mask_special(dy->label, &mask_flag);		/* get the mask array in special mode */
	vr = (float *) _fmalloc(sizeof(float) * npts);	/* used for the "sigma" array */
	kfillv_s(vr, (float) 1.0, npts);  // set all to 1.0 for now

    for(i = 0; i < m; i++) lista[i] = i;	// store indexices of variables
    for(i = 0; i < m; i++) a[i] = (float) simp[0][i];  // get starting values
    alambda = (float) -1.0;
    for(niter = 0; niter < maxiter; niter++) {
		if(keyst()) {
			recalc_flag=save_flag;/* restore the recalc flag */
			goto marq_done;
		}
		if(!(niter % ndisp))	{
			if(niter > 0) erase_dial();
	   		mvprint(1,ERRLIN+1,"-- (mrq) Iteration %5d --",niter);
		}
		mrqmin_s(vx, vd, vr, npts, &a[0], &lista[0], m, m, &covar[0][0], &alpha1[0][0], 20,20, &chisq, &alambda, mrq_fnc);

		if(!(niter % ndisp))	{
			for(j = 0; j < m; j++) mvprint(1,-1,"n[%1d]    %8.3g  ",j,a[j]);
			mvprint(1,-1,"chisq:    %8.3g     alambda=%12.4e", chisq, alambda); /* show the status */
		}
		if((ochisq-chisq) > 0 && (ochisq-chisq)/chisq < maxer) break; // terminal conditions
		ochisq = chisq;
	}
	alambda = (float) 0.0;  // must call one more time
	mrqmin_s(vx, vd, vr, npts, &a[0], &lista[0], m, m, &covar[0][0], &alpha1[0][0], 20,20, &chisq, &alambda, mrq_fnc);
	erase_dial();
	mvprint(1,ERRLIN+1, "Final results: (niter = %d)", niter);
	for(j = 0; j < m; j++) mvprint(1,-1,"n[%1d]    %8.3g  Err: %12.4e",j,a[j], sqrt(fabs(covar[j][j])));
	mvprint(1,-1,"chisq:    %8.3g ", chisq); /* show the status */
// store results of variables and fit
    for(j = 0; j < m; j++) var[j] = a[j];
	command(py->func,2);/* recompute the theoretical equation */
    var[m] = chisq;  // store the chiseq value
	var[m+1] = niter;  // store total number of iterations for user test
	for(i = 0; i < m; i++) simp[0][i] = a[i];	// store results for continued iterations!
marq_done:
	_ffree(vd);	/* free the temporary data array */
	_ffree(vb);	/* free the temporary mask array */
	_ffree(vr);	/* free the temporary result array */
	_ffree(vx);
	recalc_flag=save_flag;/* restore the recalc_flag */
	return(0);
}

/*
	-------------------------------------------------------

		Simplex function

	-------------------------------------------------------
*/


static int simplex_fit(void)
{
	int nerr_condx = 0;
	int i, j;
	erase_dial();
	npts = size_strip_pt(dy);
	vd = load_array(dy->label); /* raw data */
	vb = load_mask_special(dy->label, &mask_flag);		/* get the mask array in special mode */
	vr = _fmalloc(sizeof(float) * npts);	/* result temp space! */
	sum_res(simp[0]);/* first vertex */
//	maxerr[m] = 0.0;
	for(i = 0 ; i < m ; i++) { /*	compute the offset of the vertex of the starting simplex */
		p[i]=step[i] * (sqrt((double) n) + m -1) /((double)m*ROOT2);
		q[i]=step[i] * (sqrt((double) n) -1) / ((double)m*ROOT2);
	}
	for(i = 0 , nerr_condx = 0; i < n ; i++) if(maxerr[i] > 0.0) nerr_condx++;
	for(i = 1; i < n ; i++) { /*	all vertex of the starting simplex */
		if(keyst()) break;/* abort the function with report */
		for(j = 0 ; j < m ; j++) simp[i][j] = simp[0][j] + q[j];
		simp[i][i-1] = simp[0][i-1] + p[i-1];
		sum_res(simp[i]);/* and their residuals */
	}
	/*preset */
	for(i = 0; i < N; i++) l[i] = h[i] = 1;
	order();
	first();
	erase_dial();
	/*	----- iteration loop ------	*/
	for(niter = 0 ; niter < maxiter ; niter++) {
		if(keyst()) goto simp_done;
		for(i = 0; i < n; i++) center[i] = 0.;
		for(i = 0; i < n; i++) { /* compute the centroid */
			if( i != h[m] ) { /* excluding the worst */
				for( j = 0; j < m ; j++)
				center[j] += simp[i][j];	/* sums here */
			}
		}
		for(i = 0; i < n; i++) { /* first attempt to reflect	*/
			center[i] = center[i] / (double)m;	/* mean (centroid) here ) */
			next[i] = (1.0+ALFA)*center[i]-ALFA*simp[h[m]][i];
			/* new vertex is the specular reflection of the worst */
		}
		sum_res(next);
		if(next[m] <= simp[l[m]][m]) { /* better than the best ? */
			new_vertex();/*	accepted */
			for(i = 0 ; i < m ; i++)
				next[i]= GAMMA * simp[h[m]][i]+(1.0-GAMMA) * center[i];
			sum_res(next);/* still better ? */
			if(next[m] <= simp[l[m]][m]) new_vertex();
		} /* expansion accepted */
		else { /*if not better than the best */
			if(next[m] <= simp[h[m]][m]) new_vertex();/* better, worst */
			else  { /* worst than worst - contract */
				for(i = 0 ; i < m ; i++)
					next[i] = BETA * simp[h[m]][i]+(1.0-BETA) * center[i];
				sum_res(next);
				if(next[m] <= simp[h[m]][m]) new_vertex();/* contraction ok*/
				else { /* if still bad shrink all bad vertex */
					for(i = 0; i < n; i++) {
						for(j = 0 ; j < m ; j++)
							simp[i][j]=BETA*(simp[i][j] +simp[l[m]][j]);
						sum_res(simp[i]);
					}
				}
			}
		}
		order();
		for(j=0, done=0; j < n ;j++) {
				if(simp[h[j]][j] == 0.0) error[j] = 1.0;	/* avoid division by 0 */
				else error[j]=fabs((simp[h[j]][j] - simp[l[j]][j])/simp[h[j]][j]);
				if(maxerr[j] > 0.0) done += (maxerr[j] > error[j]); /* when we get below a particular limit, we set the flag */
		}
//		mvprint(1,20,"Ncount: %d",ncount++);
		if((nerr_condx > 0) && (done == nerr_condx)) break; /* all set limits have been reached */
	} /* main iteration	*/
	LOOP { /* average each parameter */
		for(mean[i]=0. , j=0 ; j < n; j++)
			mean[i] += simp[j][i];
		mean[i] = mean[i] / (double) n;
	}
	report();
simp_done:
	_ffree(vd);	/* free the temporary data array */
	_ffree(vb);	/* free the temporary mask array */
	_ffree(vr);	/* free the temporary result array */
	recalc_flag=save_flag;/* restore the recalc_flag */
	return(0);/*	ok done	*/
}
