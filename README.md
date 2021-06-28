# pronoun-interpretation-code
Code related to Bayesian model variants for child and adult pronoun interpretation behavior

### Readme file for pronouninterpretationmodels.py python code, 
which is used to
    implement various theories (models) of the underlying mechanisms for observable
    pronoun interpretation behavior in experiments with children
    and adults.

### Updated by Lisa Pearl, June 2021 ###

*If using this code, please cite the following paper:*

Pearl, Lisa & Forsythe, Hannah. 2021. Manuscript. Inaccurate representations, inaccurate deployment, or both?
Using computational cognitive modeling to investigate the development of pronoun interpretation in Spanish.
University of California, Irvine.

*Note: The pronouninterpretationmodels.py code requires the numpy and pandas python libraries.*


****************************
## pronounintepretationmodels.py code ##
****************************
## (1) Basic usage ##

Put the pronouninterpretationmodels.py file in a directory of your choice. 
Navigate to that directory and run the code. 

If your input files are the input/corpusprobabilities.txt file (for the corpus probabilities) 
and the input/exptobserved.txt file (for the participant data from the experiments), 
you could do it this way from the command line and get the default options for all the optional parameter 
values below (only do age 3, run all sigmas, 0.1 < sigma < 4.0 by .1 increments; 
run all betas 0 < beta < 1.0 by .1 increments; baseline model).

    python pronouninterpretationmodels.py 'input/corpusprobabilities.txt' 'input/exptobserved.txt'

You can use the -h option to view a help file that describes all the different options and their default values.

    python pronouninterpretationmodels.py -h

****************************
## positional arguments: ##


    corpusprob_file       input file containing corpus probabilities
    exptobs_file          input file containing observed experiment responses

****************************
## optional arguments: ##

    -h, --help
    show this help message and exit

    --output_file OUTPUT_FILE, -o OUTPUT_FILE
    output file name
  
    --model_type MODEL_TYPE, -mt MODEL_TYPE
    model type 1=baseline, 2=inaccurate representations, 3=inaccurate deployment, 4=inaccurate both

### prior parameters ###

    --sigma_prior_low SIGMA_PRIOR_LOW, -spl SIGMA_PRIOR_LOW
    lower bound for sigma prior parameter
    --sigma_prior_high SIGMA_PRIOR_HIGH, -sph SIGMA_PRIOR_HIGH
    upper bound for sigma prior parameter
    --sigma_prior_inc SIGMA_PRIOR_INC, -spi SIGMA_PRIOR_INC
    increment for sigma prior parameter
  
    --beta_prior_low BETA_PRIOR_LOW, -bpl BETA_PRIOR_LOW
    lower bound for beta prior parameter
    --beta_prior_high BETA_PRIOR_HIGH, -bph BETA_PRIOR_HIGH
    upper bound for beta prior parameter
    --beta_prior_inc BETA_PRIOR_INC, -bpi BETA_PRIOR_INC
    increment for beta prior parameter
 
### pronoun form parameters ###
 
    --sigma_form_low SIGMA_FORM_LOW, -sfl SIGMA_FORM_LOW
    lower bound for sigma form parameter
    --sigma_form_high SIGMA_FORM_HIGH, -sfh SIGMA_FORM_HIGH
    upper bound for sigma form parameter
    --sigma_form_inc SIGMA_FORM_INC, -sfi SIGMA_FORM_INC
    increment for sigma form parameter

    --beta_form_low BETA_FORM_LOW, -bfl BETA_FORM_LOW
    lower bound for beta form parameter
    --beta_form_high BETA_FORM_HIGH, -bfh BETA_FORM_HIGH
    upper bound for beta form parameter
    --beta_form_inc BETA_FORM_INC, -bfi BETA_FORM_INC
    increment for beta form parameter
  
### connective parameters ###

    --sigma_con_low SIGMA_CON_LOW, -scl SIGMA_CON_LOW
    lower bound for sigma connective parameter
    --sigma_con_high SIGMA_CON_HIGH, -sch SIGMA_CON_HIGH 
    upper bound for sigma connective parameter
    --sigma_con_inc SIGMA_CON_INC, -sci SIGMA_CON_INC
    increment for sigma connective parameter
  
    --beta_con_low BETA_CON_LOW, -bcl BETA_CON_LOW
    lower bound for beta connective parameter
    --beta_con_high BETA_CON_HIGH, -bch BETA_CON_HIGH
    upper bound for beta connective parameter 
    --beta_con_inc BETA_CON_INC, -bci BETA_CON_INC
    increment for beta connective parameter

### morphology parameters ###
  
    --sigma_mor_low SIGMA_MOR_LOW, -sml SIGMA_MOR_LOW
    lower bound for sigma morphology parameter
    --sigma_mor_high SIGMA_MOR_HIGH, -smh SIGMA_MOR_HIGH
    upper bound for sigma morphology parameter
    --sigma_mor_inc SIGMA_MOR_INC, -smi SIGMA_MOR_INC
    increment for sigma morphology parameter

    --beta_mor_low BETA_MOR_LOW, -bml BETA_MOR_LOW
    lower bound for beta morphology parameter
    --beta_mor_high BETA_MOR_HIGH, -bmh BETA_MOR_HIGH
    upper bound for beta morphology parameter
    --beta_mor_inc BETA_MOR_INC, -bmi BETA_MOR_INC
    increment for beta morphology parameter

### for model variants with multiple options (2,3,4) ###

    --topn TOPN, -tn TOPN
    number of top parameter settings stored

### age-specific, rather than all ages at once ###

    --only_one_age, -ooa  
    do only for one age specified by age parameter
    --age AGE, -a AGE     
    do only for one age specified by age parameter in string format like "3" or "adult"

**********************
## example full usage for each model type ##

*baseline (model type 1)*

    python3 -u pronouninterpretationmodels.py 'input/corpusprobabilities.txt' 'input/exptobserved.txt' -o 'baseline.out' -mt 1

*inaccurate representations (model type 2, with sigmas)*

    python3 -u pronouninterpretationmodels.py 'input/corpusprobabilities.txt' 'input/exptobserved.txt' -o 'inaccrep.out' -mt 2 -spl 0.2 -sph 0.4 -spi 0.2 -sfl 0.1 -sfh 0.3 -sfi 0.2 -scl 0.3 -sch 0.6 -sci 0.3 -sml 0.4 -smh 0.7 -smi 0.3 -tn 5 -ooa -a "3"

*inaccurate deployment (model type 3, with betas)*

    python3 -u pronouninterpretationmodels.py 'input/corpusprobabilities.txt' 'input/exptobserved.txt' -o 'inaccdepl.out' -mt 3 -bpl 0.01 -bph 0.2 -bpi 0.2 -bfl 0.01 -bfh 0.4 -bfi 0.2 -bcl 0.01 -bch 0.3 -bci 0.3 -bml 0.01 -bmh 0.8 -bmi 0.4  -tn 5 -ooa -a "5"

*both inaccurate (model type 4, with sigmas and betas)*

    python3 -u pronouninterpretationmodels.py 'input/corpusprobabilities.txt' 'input/exptobserved.txt' -o 'bothinacc.out' -mt 4 -spl 0.2 -sph 0.2 -spi 0.2 -sfl 0.1 -sfh 0.3 -sfi 0.2 -scl 0.3 -sch 0.6 -sci 0.3 -sml 0.4 -smh 0.6 -smi 0.2 -bpl 0.0 -bph 0.2 -bpi 0.2 -bfl 0.0 -bfh 0.5 -bfi 0.5 -bcl 0.6 -bch 0.9 -bci 0.3 -bml 0.4 -bmh 0.8 -bmi 0.2 -tn 5 -ooa -a "adult"



**********************
## (2) Expected input files ##

(a) corpus probabilities (priors and likelihoods) (as in input/corpusprobabilities.txt)

(b) participant response summaries from pronoun interpretation experiments (as in input/exptobserved.txt)

*********************
## (3) Output file information ##

    Corpus probabilities input file is *corpus probabilities input file*
    Observed expt responses input file is *observed participant responses input file*
    Output file is *output file*
    Model type is *model type number, 1-4*
    *sigma_prior_low* <= sigma_prior <= *sigma_prior_high* by *sigma_prior_increment*
    *sigma_form_low* <= sigma_form <= *sigma_form_high* by *sigma_form_increment*
    *sigma_con_low* <= sigma_con <= *sigma_con_high* by *sigma_con_increment*
    *sigma_mor_low* <= sigma_mor <= *sigma_mor_high* by *sigma_mor_increment*
    *beta_prior_low* <= beta_prior <= *beta_prior_high* by *beta_prior_increment*
    *beta_form_low* <= beta_form <= *beta_form_high* by *beta_form_increment*
    *beta_con_low* <= beta_con <= *beta_con_high* by *beta_con_increment*
    *beta_mor_low* <= beta_mor <= *beta_mor_high* by *beta_mor_increment*
    Top *top_n* model parameter settings stored
    Only doing age *age*
    Corpus prior probabilities:
    anteNonPl: 0.1286089
    ...
    anteSubjSg: 0.3622047
    Corpus likelihood probabilities:
    DespNonPl: 0.393939394
    ...
    SgSubjSg: 0.998411017

*(info from observed experimental participants file)*

    age connective form morphology subj_morph nonsubj_morph num_subj_resp num_nonsubj_resp
    3 despues null sg sg pl 22 39
    ...
    adult porque null pl sg pl 2 18
    Model type *model_number*: *model description*
    *sigma and beta ranges*
    ...
    Getting top *top_n* options
    M for BIC calculation is *appropriate_M* 
(1 for baseline, 4 for inaccurate representations or inaccurate deployment, 8 for both inaccurate)

    Only doing age *age*
    Model type *model type*: *model description*, m = *appropriate_M*
    age *age* top *top_n* logprob and BICs

*(for baseline, model type 1)*

    age *age* logprob = *age logprob*, age 3 BIC = *age BIC*


*(for inaccurate representations, model type 2)*

           sigma_prior	sigma_for	sigma_con	sigma_mor		logprob	        BIC

    0	*val1*	        *val2*	        *val3*	        *val4*	    *logprob*	*BIC*

    ...

    topn-1	*val1*	        *val2*	        *val3*	        *val4*	    *logprob*	*BIC*

	sigma_prior: low-high range in topn
	sigma_for: low-high range in topn
	sigma_con: low-high range in topn
	sigma_mor: low-high range in topn

*(for inaccurate representations, model type 3)*

           beta_prior	beta_for	beta_con	beta_mor		logprob	        BIC

    0	*val1*	        *val2*	        *val3*	        *val4*	    *logprob*	*BIC*

    ...

    topn-1	*val1*	        *val2*	        *val3*	        *val4*	    *logprob*	*BIC*

	beta_prior: low-high range in topn
	beta_for: low-high range in topn
	beta_con: low-high range in topn
	beta_mor: low-high range in topn

*(for both inaccurate, model type 4)*

           sigma_prior	sigma_for	sigma_con	sigma_mor	beta_prior	beta_for	beta_con	beta_mor	logprob	        BIC

    0	*val1*	        *val2*	        *val3*	        *val4*	        *val5*	        *val6*	        *val7*	        *val8*	        *logprob*	*BIC*

    ...

    topn-1	*val1*	        *val2*	        *val3*	        *val4*	        *val5*	        *val6*	        *val7*	        *val8*	        *logprob*	*BIC*

	sigma_prior: low-high range in topn
	sigma_for: low-high range in topn
	sigma_con: low-high range in topn
	sigma_mor: low-high range in topn
	beta_prior: low-high range in topn
	beta_for: low-high range in topn
	beta_con: low-high range in topn
	beta_mor: low-high range in topn
