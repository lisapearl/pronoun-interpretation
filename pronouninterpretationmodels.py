#!/usr/bin/python3
""" Code for calculating model predictions for pronoun interpretations,
    using Bayesian inference framework
    Initially created May 2021"""

""" 
Goal: Given 
(a) corpus estimates of prior and likelihood probabilities
      for different antecedent types and feature values for
      pronoun form, discourse connective, and agreement morphology,
and
(b) counts of subject responses from pronoun interpretation experiments,
then
calculate (best) log probability and BIC for different model types:
        (i) baseline Bayes
        (ii) inaccurate representations only
        (iii) inaccurate deployment only
        (iv) both inaccurate representations and inaccurate deployment.
This involves identifying the best-fitting parameters for the inaccurate model types
(sigmas for inaccurate representations, betas for inaccurate deployment).
"""

"""
Needed: 
(a) Corpus estimates of relevant probabilities (e.g., as in input/corpusprobabilities.txt)
(b) Observed participant responses to experimental stimuli (e.g., as in input/exptobserved.txt)
"""

"""
General outline of code:
(1) Read in corpus estimates of relevant probabilities and store these.
- priors on antecedents: anteSubjSg, anteSubjPl, anteNonSubjSg, anteNonSubjPl
- likelihoods given certain values (24 of these - 6 for each of SubjSg, SubjPl, NonSubjSg, NonSubjPl):
  form: nullGivAnteSubjSg, overtGivAnteSubjSg, ...
  connective: despuesGivAnteSubjSg, despuesGivAnteSubjPl, ...
  morph: sgGivAnteSubjSg, plGivAntSubjPl, ... 
(2) Read in observed participant responses and store these.
- of the form: age,connective,form,morphology,subj_mor,nonsubj_mor,num_subj,num_nonsubj
- each age has collection of 16 conditions associated, 
   involving feature values (form, connective, morph), morphology on subject and non-subject,
   and number of responses for subject vs. non-subject for that condition.
(3) Determine which model type to calculate likelihood and BIC for.
- if baseline Bayes, no parameter search required
- else need to do search between lower_sigma <= sigma <= upper_sigma, lower_beta <= beta <= upper_beta
(4) If parameter search, set up parameter search loops and storage of top topNStored
(5) For each model instance (baseline Bayes, or particular parameter values for inaccurate variants)
- calculate log probability of observed data, given individual posteriors (i.e., probability of responses | parameters)
- if parameter search, store topNStored best log probabilities with their parameter values
(6) Print out (best) log probability + parameter values if needed, and calculate BIC.
- BIC free parameter # k = 1 for baseline, 4 for inaccurate representations and inaccurate deployment, 
  8 for inaccurate both
"""

import argparse # for reading in command line arguments
import re # for regular expressions
import sys # for stdout and stderr printing
import pandas as pd # for using dataframes
import math # for log
import numpy as np # for numpy stuff

def main():
    #print("Sanity check")
    # read in relevant argument values from the command line
    args = read_arguments()
    # let's make sure these values are what we expect
    print('Corpus probabilities input file is {}'.format(args.corpusprob_file))
    print('Observed expt responses input file is {}'.format(args.exptobs_file))
    print('Output file is {}'.format(args.output_file))
    print('Model type is {}'.format(args.model_type))
    if (args.model_type == 2) or (args.model_type == 4):
        print('{} <= sigma_prior <= {} by {}'.format(args.sigma_prior_low, args.sigma_prior_high, args.sigma_prior_inc))
        print('{} <= sigma_form <= {} by {}'.format(args.sigma_form_low, args.sigma_form_high, args.sigma_form_inc))
        print('{} <= sigma_con <= {} by {}'.format(args.sigma_con_low, args.sigma_con_high, args.sigma_con_inc))
        print('{} <= sigma_mor <= {} by {}'.format(args.sigma_mor_low, args.sigma_mor_high, args.sigma_mor_inc))
    if (args.model_type == 3) or (args.model_type == 4):
        print('{} <= beta_prior <= {} by {}'.format(args.beta_prior_low, args.beta_prior_high, args.beta_prior_inc))
        print('{} <= beta_form <= {} by {}'.format(args.beta_form_low, args.beta_form_high, args.beta_form_inc))
        print('{} <= beta_con <= {} by {}'.format(args.beta_con_low, args.beta_con_high, args.beta_con_inc))
        print('{} <= beta_mor <= {} by {}'.format(args.beta_mor_low, args.beta_mor_high, args.beta_mor_inc))
    print('Top {} model parameter settings stored'.format(args.topn))
    if(args.only_one_age):
        print('Only doing age {}'.format(args.age))

    # open output_file for writing
    output_file = open(args.output_file, 'w')
    # let's write this information out to the output file for our records
    output_file.write("Corpus probabilities input file is {}\n".format(args.corpusprob_file))
    output_file.write("Observed expt responses input file is {}\n".format(args.exptobs_file))
    output_file.write("Output file is {}\n".format(args.output_file))
    output_file.write("Model type is {}\n".format(args.model_type))
    if (args.model_type == 2) or (args.model_type == 4):
        output_file.write("{} <= sigma_prior <= {} by {}\n".format(args.sigma_prior_low, args.sigma_prior_high, args.sigma_prior_inc))
        output_file.write("{} <= sigma_form <= {} by {}\n".format(args.sigma_form_low, args.sigma_form_high, args.sigma_form_inc))
        output_file.write("{} <= sigma_con <= {} by {}\n".format(args.sigma_con_low, args.sigma_con_high, args.sigma_con_inc))
        output_file.write("{} <= sigma_mor <= {} by {}\n".format(args.sigma_mor_low, args.sigma_mor_high, args.sigma_mor_inc))
    if (args.model_type == 3) or (args.model_type == 4):
        output_file.write("{} <= beta_prior <= {} by {}\n".format(args.beta_prior_low, args.beta_prior_high, args.beta_prior_inc))
        output_file.write("{} <= beta_form <= {} by {}\n".format(args.beta_form_low, args.beta_form_high, args.beta_form_inc))
        output_file.write("{} <= beta_con <= {} by {}\n".format(args.beta_con_low, args.beta_con_high, args.beta_con_inc))
        output_file.write("{} <= beta_mor <= {} by {}\n".format(args.beta_mor_low, args.beta_mor_high, args.beta_mor_inc))
    output_file.write("Top {} model parameter settings stored\n".format(args.topn))
    if(args.only_one_age):
        output_file.write("Only doing age {}\n".format(args.age))

    # Read in corpus estimates of priors and likelihoods
    # want to put these into corpus probabilities structure
    this_corpusprobabilities = CorpusProbabilities(args.corpusprob_file)
    # print these out
    print("Read in corpus probabilities.")
    # output file version
    this_corpusprobabilities.print_corpus_info(output_file)

    # Read in observed experiment responses
    # want to put these into observed experiment response structure
    this_obsexptresp = ObservedExptResponses(args.exptobs_file)
    print("Read in observed experiment responses.")
    this_obsexptresp.print_obsexptresp(output_file)

    # Added in only_one_age and age restrictions, if relevant (set options in ModelVariant)
    # Determine which model type, and if parameter search is needed
    if args.model_type == 1: # baseline
        this_model = ModelVariant(1)
    elif args.model_type == 2: # inacc rep
        this_model = ModelVariant(2,
                                  args.sigma_prior_low, args.sigma_prior_high, args.sigma_prior_inc,
                                  args.sigma_form_low, args.sigma_form_high, args.sigma_form_inc,
                                  args.sigma_con_low, args.sigma_con_high, args.sigma_con_inc,
                                  args.sigma_mor_low, args.sigma_mor_high, args.sigma_mor_inc,
                                  top_n=args.topn)
    elif args.model_type == 3: # inacc depl
        this_model = ModelVariant(3,
                                  beta_prior_low=args.beta_prior_low, beta_prior_high=args.beta_prior_high, beta_prior_inc=args.beta_prior_inc,
                                  beta_form_low=args.beta_form_low, beta_form_high=args.beta_form_high, beta_form_inc=args.beta_form_inc,
                                  beta_con_low=args.beta_con_low, beta_con_high=args.beta_con_high, beta_con_inc=args.beta_con_inc,
                                  beta_mor_low=args.beta_mor_low, beta_mor_high=args.beta_mor_high, beta_mor_inc=args.beta_mor_inc,
                                  top_n=args.topn)
    elif args.model_type == 4:  # inacc both
        this_model = ModelVariant(4,
                                  args.sigma_prior_low, args.sigma_prior_high, args.sigma_prior_inc,
                                  args.sigma_form_low, args.sigma_form_high, args.sigma_form_inc,
                                  args.sigma_con_low, args.sigma_con_high, args.sigma_con_inc,
                                  args.sigma_mor_low, args.sigma_mor_high, args.sigma_mor_inc,
                                  args.beta_prior_low, args.beta_prior_high, args.beta_prior_inc,
                                  args.beta_form_low, args.beta_form_high, args.beta_form_inc,
                                  args.beta_con_low, args.beta_con_high, args.beta_con_inc,
                                  args.beta_mor_low, args.beta_mor_high, args.beta_mor_inc,
                                  args.topn)

    # if args.only_one_age, update model settings to reflect this
    if args.only_one_age:
        this_model.setonlyage(args.age)

    print("debug: this model's settings")
    this_model.print_modelsettings()
    this_model.print_modelsettings(output_file)

    # calculate (best) log probability and BIC for observed experiment responses for each age group
    this_model.calculatelogprobandBIC(this_corpusprobabilities, this_obsexptresp)
    # debug
    print("debug: this model's (best) logprob and BIC")
    # model variants with top_n
    this_model.print_logprobandBIC()
    this_model.print_logprobandBIC(output_file)

    # close output file
    output_file.close()
#####################
class ModelVariant:
    # initialize with relevant information for the model variant
    def __init__(self, model_type,
                 sigma_prior_low=1.0, sigma_prior_high=1.0, sigma_prior_inc=1.0, # sigmas for inacc-rep and both-inacc; default=1.0 for not used
                 sigma_form_low=1.0, sigma_form_high=1.0, sigma_form_inc=1.0, # sigmas for inacc-rep and both-inacc; default=1.0 for not used
                 sigma_con_low=1.0, sigma_con_high=1.0, sigma_con_inc=1.0, # sigmas for inacc-rep and both-inacc; default=1.0 for not used
                 sigma_mor_low=1.0, sigma_mor_high=1.0, sigma_mor_inc=1.0, # sigmas for inacc-rep and both-inacc; default=1.0 for not used
                 beta_prior_low=1.0, beta_prior_high=1.0, beta_prior_inc=1.0, # sigmas for inacc-rep and both-inacc; default=1.0 for not used
                 beta_form_low=1.0, beta_form_high=1.0, beta_form_inc=1.0, # sigmas for inacc-rep and both-inacc; default=1.0 for not used
                 beta_con_low=1.0, beta_con_high=1.0, beta_con_inc=1.0, # sigmas for inacc-rep and both-inacc; default=1.0 for not used
                 beta_mor_low=1.0, beta_mor_high=1.0, beta_mor_inc=1.0, # sigmas for inacc-rep and both-inacc; default=1.0 for not used
                 top_n=1, # for inacc variants that do parameter search
                 only_one_age=False, # default not to just do one age, but to do all of them
                 which_age = "3" # default age is only_one_age is true
                 ):
        self.model_type = model_type
        if self.model_type == 1:
            # baseline -- just calculate log probability with true probabilities once
            self.m = 0 # m = 0 for BIC
        elif self.model_type == 2:
            self.m = 4 # m = 4 for BIC
        elif self.model_type == 3:
            self.m = 4 # m = 4 for BIC
        else:
            self.m = 8 # m = 8 for BIC
        # prior
        self.sigma_prior_low = sigma_prior_low
        self.sigma_prior_high = sigma_prior_high
        self.sigma_prior_inc = sigma_prior_inc
        self.beta_prior_low = beta_prior_low
        self.beta_prior_high = beta_prior_high
        self.beta_prior_inc = beta_prior_inc
        # form
        self.sigma_form_low = sigma_form_low
        self.sigma_form_high = sigma_form_high
        self.sigma_form_inc = sigma_form_inc
        self.beta_form_low = beta_form_low
        self.beta_form_high = beta_form_high
        self.beta_form_inc = beta_form_inc
        # con
        self.sigma_con_low = sigma_con_low
        self.sigma_con_high = sigma_con_high
        self.sigma_con_inc = sigma_con_inc
        self.beta_con_low = beta_con_low
        self.beta_con_high = beta_con_high
        self.beta_con_inc = beta_con_inc
        # mor
        self.sigma_mor_low = sigma_mor_low
        self.sigma_mor_high = sigma_mor_high
        self.sigma_mor_inc = sigma_mor_inc
        self.beta_mor_low = beta_mor_low
        self.beta_mor_high = beta_mor_high
        self.beta_mor_inc = beta_mor_inc
        self.top_n = top_n
        self.only_one_age = only_one_age
        # if only_one_age = True, then set which age, then update execution below
        if(self.only_one_age):
            self.which_age = which_age

    def setonlyage(self, which_age):
        self.only_one_age = True
        self.which_age = which_age

    def calculatelogprobandBIC(self, corpusprobs, obsexptprobs):
        # calculate (best top N) log prob and BIC for observed experimental data
        if self.model_type == 1:
            # baseline -- just calculate true log prob, so just a single num
            # (no need for a data frame)
            priors = corpusprobs.get_priors()
            likelihoods = corpusprobs.get_likelihoods()

            print("debug: calculating true log prob for baseline model")
            if (not self.only_one_age) or ((self.only_one_age) and (self.which_age == "3")):
                self.age_3_logprob = self.calculatelogprobforage(3, priors, likelihoods, obsexptprobs)
                print("debug, self.age_3_logprob = {}".format(self.age_3_logprob))
                # the only one, so go ahead and calculate the BIC
                self.age_3_BIC =self.calculateBICforage(3, self.age_3_logprob, obsexptprobs, self.m)
                print("debug, self.age_3_BIC = {}".format(self.age_3_BIC))

            if (not self.only_one_age) or ((self.only_one_age) and (self.which_age == "4")):
                self.age_4_logprob = self.calculatelogprobforage(4, priors, likelihoods, obsexptprobs)
                print("debug, self.age_4_logprob = {}".format(self.age_4_logprob))
                # the only one, so go ahead and calculate the BIC
                self.age_4_BIC =self.calculateBICforage(4, self.age_4_logprob, obsexptprobs, self.m)
                print("debug, self.age_4_BIC = {}".format(self.age_4_BIC))

            if (not self.only_one_age) or ((self.only_one_age) and (self.which_age == "5")):
                self.age_5_logprob = self.calculatelogprobforage(5, priors, likelihoods, obsexptprobs)
                print("debug, self.age_5_logprob = {}".format(self.age_5_logprob))
                # the only one, so go ahead and calculate the BIC
                self.age_5_BIC =self.calculateBICforage(5, self.age_5_logprob, obsexptprobs, self.m)
                print("debug, self.age_5_BIC = {}".format(self.age_5_BIC))

            if (not self.only_one_age) or ((self.only_one_age) and (self.which_age == "adult")):
                self.age_adult_logprob = self.calculatelogprobforage("adult", priors, likelihoods, obsexptprobs)
                print("debug, self.age_adult_logprob = {}".format(self.age_adult_logprob))
                # the only one, so go ahead and calculate the BIC
                self.age_adult_BIC =self.calculateBICforage("adult", self.age_adult_logprob, obsexptprobs, self.m)
                print("debug, self.age_adult_BIC = {}".format(self.age_adult_BIC))

        # other model types that require top_n
        else:
            #        then also need to do the parameter search for the other model types,
            #        with dataframe of topN values returned
            # initialize top_n data frame to hold best parameter value combinations for age
            if (not self.only_one_age) or ((self.only_one_age) and (self.which_age == "3")):
                # age 3
                self.age_3_top_n = self.get_top_n("3", corpusprobs, obsexptprobs)
                # once have (best top N) log probs, then calculate BIC(s)
                self.age_3_top_n["BIC"] = self.calculateBICforage(3,self.age_3_top_n["logprob"],obsexptprobs, self.m)
                print("debug, age 3 top {} logprobs with BICs".format(self.top_n))
                print(self.age_3_top_n)
                sys.stderr.write("age 3 done\n")  # for charting progress

            if (not self.only_one_age) or ((self.only_one_age) and (self.which_age == "4")):
                self.age_4_top_n = self.get_top_n("4", corpusprobs, obsexptprobs)
                # once have (best top N) log probs, then calculate BIC(s)
                self.age_4_top_n["BIC"] = self.calculateBICforage(4,self.age_4_top_n["logprob"],obsexptprobs, self.m)
                print("debug, age 4 top {} logprobs with BICs".format(self.top_n))
                print(self.age_4_top_n)
                sys.stderr.write("age 4 done\n")  # for charting progress

            if (not self.only_one_age) or ((self.only_one_age) and (self.which_age == "5")):
                self.age_5_top_n = self.get_top_n("5", corpusprobs, obsexptprobs)
                # once have (best top N) log probs, then calculate BIC(s)
                self.age_5_top_n["BIC"] = self.calculateBICforage(5,self.age_5_top_n["logprob"],obsexptprobs, self.m)
                print("debug, age 5 top {} logprobs with BICs".format(self.top_n))
                print(self.age_5_top_n)
                sys.stderr.write("age 5 done\n")  # for charting progress

            if (not self.only_one_age) or ((self.only_one_age) and (self.which_age == "adult")):
                self.age_adult_top_n = self.get_top_n("adult", corpusprobs, obsexptprobs)
                # once have (best top N) log probs, then calculate BIC(s)
                self.age_adult_top_n["BIC"] = self.calculateBICforage("adult",self.age_adult_top_n["logprob"],obsexptprobs, self.m)
                print("debug, age adult top {} logprobs with BICs".format(self.top_n))
                print(self.age_adult_top_n)
                sys.stderr.write("age adult done\n")  # for charting progress

    # parameter search
    def get_top_n(self, age, corpusprobs, obsexptprobs):
        # initialize empty data frame to hold topN
        age_top_n = pd.DataFrame()
        if self.model_type == 2:
            # just sigmas
            # columns: sigma_prior, sigma_for, sigma_con, sigma_mor, logprob
            age_top_n = pd.DataFrame(columns=['sigma_prior', 'sigma_for', 'sigma_con', 'sigma_mor', 'logprob'])

            # loop through sigmas
            for sigma_prior in np.arange(self.sigma_prior_low, self.sigma_prior_high + self.sigma_prior_inc, self.sigma_prior_inc):
                for sigma_for in np.arange(self.sigma_form_low, self.sigma_form_high + self.sigma_form_inc, self.sigma_form_inc):
                    sys.stderr.write(".")  # for charting progress
                    for sigma_con in np.arange(self.sigma_con_low, self.sigma_con_high + self.sigma_con_inc, self.sigma_con_inc):
                        sys.stderr.write(".")  # for charting progress
                        for sigma_mor in np.arange(self.sigma_mor_low, self.sigma_mor_high + self.sigma_mor_inc, self.sigma_mor_inc):
                            # for this combination of sigmas, calculate updated priors and likelihoods from corpusprobs object
                            sigma_priors = corpusprobs.get_sigmapriors(sigma_prior)
                            # get updated likelihoods (should be normalized by pairs: e.g., null vs overt given antecedent type)
                            sigma_likelihoods = corpusprobs.get_sigmalikelihoods(sigma_for, sigma_con, sigma_mor)
                            # now get log probability for age with these priors and likelihoods
                            this_logprobforage = self.calculatelogprobforage(age, sigma_priors, sigma_likelihoods, obsexptprobs)
                            # see if it's better than top_n; if so, add; if not, ignore
                            # check size of dataframe
                            top_n_size = len(age_top_n.index)
                            if (top_n_size < self.top_n):
                                # add it
                                age_top_n.loc[top_n_size] = [sigma_prior, sigma_for, sigma_con, sigma_mor, this_logprobforage]
                            else:
                                # see if the log prob is better than the ones already there
                                worst_logprob = age_top_n.at[top_n_size-1,'logprob']
                                if this_logprobforage > worst_logprob:
                                    print("debug, found better sigmas (prior = {}, for = {}, con = {}, mor = {}) with logprob {}".format(sigma_prior, sigma_for, sigma_con, sigma_mor, this_logprobforage))
                                    # drop previous worst
                                    age_top_n.drop([top_n_size-1])
                                    # add new one
                                    age_top_n.loc[top_n_size-1] = [sigma_prior, sigma_for, sigma_con, sigma_mor, this_logprobforage]
                            # resort on logprob
                            age_top_n.sort_values(by=['logprob'], inplace=True, ignore_index=True, ascending=False)

        elif self.model_type == 3:
            print("debug: inaccurate deployment with betas")
            # just betas
            # DONE: Setup by item information in np arrays that can be quickly combined with current betas
            #        Will need to update all the dependent calculations to use this array.
            priors = corpusprobs.get_priors()
            likelihoods = corpusprobs.get_likelihoods()
            # this is a numpy array of logprobs, with each item having a row (in order from obsexptprobs)
            # each column in order by betas from paper
            # columns: useall,
            #          not(prior), not(form), not(form, prior),
            #          not(con), not(con, prior),
            #          not(mor), not(mor, prior)
            #          not(form, con), not(form, con, prior)
            #          not(con, mor), not(con, mor, prior)
            #          not(form, mor), not(form, mor, prior)
            #          not(form, con, mor), not(form, con, mor, prior)
            # can use it to create rapid weighted logprobs for each item, based on betas
            self.initbetaitemsubjprobsforage(age, priors, likelihoods, obsexptprobs)

            # columns: beta_prior, beta_for, beta_con, beta_mor, logprob
            age_top_n = pd.DataFrame(columns=['beta_prior', 'beta_for', 'beta_con', 'beta_mor', 'logprob'])
            # loop through betas
            for beta_prior in np.arange(self.beta_prior_low, self.beta_prior_high + self.beta_prior_inc, self.beta_prior_inc):
                for beta_for in np.arange(self.beta_form_low, self.beta_form_high + self.beta_form_inc, self.beta_form_inc):
                    for beta_con in np.arange(self.beta_con_low, self.beta_con_high + self.beta_con_inc, self.beta_con_inc):
                        sys.stderr.write(".")  # for charting progress
                        for beta_mor in np.arange(self.beta_mor_low, self.beta_mor_high + self.beta_mor_inc, self.beta_mor_inc):
                            this_logprobforage = self.calculatebetalogprobforage(age,
                                                                                 beta_prior, beta_for, beta_con, beta_mor)
                            # see if it's better than top_n; if so, add; if not, ignore
                            # check size of dataframe
                            top_n_size = len(age_top_n.index)
                            if (top_n_size < self.top_n):
                                # add it
                                age_top_n.loc[top_n_size] = [beta_prior, beta_for, beta_con, beta_mor, this_logprobforage]
                            else:
                                # see if the log prob is better than the ones already there
                                worst_logprob = age_top_n.at[top_n_size - 1, 'logprob']
                                if this_logprobforage > worst_logprob:
                                    print("debug, found better betas (prior = {}, for = {}, con = {}, mor = {}) with logprob {}".format(beta_prior, beta_for, beta_con, beta_mor, this_logprobforage))
                                    age_top_n.drop([top_n_size-1])
                                    age_top_n.loc[top_n_size-1] = [beta_prior, beta_for, beta_con, beta_mor, this_logprobforage]
                            # resort on logprob
                            age_top_n.sort_values(by=['logprob'], inplace=True, ignore_index=True, ascending=False)

        else:
            # model type 4 (both inacc) -- need sigmas and betas
            print("debug: both inaccurate model with sigmas and betas")
            # sigmas and betas
            age_top_n = pd.DataFrame(columns=['sigma_prior', 'sigma_for', 'sigma_con', 'sigma_mor',
                                              'beta_prior', 'beta_for', 'beta_con', 'beta_mor', 'logprob'])
            # loop through sigmas and betas
            for sigma_prior in np.arange(self.sigma_prior_low, self.sigma_prior_high + self.sigma_prior_inc, self.sigma_prior_inc):
                for sigma_for in np.arange(self.sigma_form_low, self.sigma_form_high + self.sigma_form_inc, self.sigma_form_inc):
                    for sigma_con in np.arange(self.sigma_con_low, self.sigma_con_high + self.sigma_con_inc, self.sigma_con_inc):
                        sys.stderr.write(".")  # for charting progress
                        for sigma_mor in np.arange(self.sigma_mor_low, self.sigma_mor_high + self.sigma_mor_inc, self.sigma_mor_inc):
                            # for this combination of sigmas, calculate updated priors and likelihoods from corpusprobs object
                            sigma_priors = corpusprobs.get_sigmapriors(sigma_prior)
                            # get updated likelihoods (should be normalized by pairs: e.g., null vs overt given antecedent type)
                            sigma_likelihoods = corpusprobs.get_sigmalikelihoods(sigma_for, sigma_con, sigma_mor)

                            # this is a numpy array of logprobs, with each item having a row (in order from obsexptprobs)
                            # each column in order by betas from paper
                            # columns: useall,
                            #          not(prior), not(form), not(form, prior),
                            #          not(con), not(con, prior),
                            #          not(mor), not(mor, prior)
                            #          not(form, con), not(form, con, prior)
                            #          not(con, mor), not(con, mor, prior)
                            #          not(form, mor), not(form, mor, prior)
                            #          not(form, con, mor), not(form, con, mor, prior)
                            # can use it to create rapid weighted logprobs for each item, based on betas
                            self.initbetaitemsubjprobsforage(age, sigma_priors, sigma_likelihoods, obsexptprobs)

                            for beta_prior in np.arange(self.beta_prior_low, self.beta_prior_high + self.beta_prior_inc, self.beta_prior_inc):
                                for beta_for in np.arange(self.beta_form_low, self.beta_form_high + self.beta_form_inc, self.beta_form_inc):
                                    for beta_con in np.arange(self.beta_con_low, self.beta_con_high + self.beta_con_inc, self.beta_con_inc):
                                        for beta_mor in np.arange(self.beta_mor_low, self.beta_mor_high + self.beta_mor_inc, self.beta_mor_inc):
                                            # for this combination of betas, calculate beta_logprob as mixture model
                                            this_logprobforage = self.calculatebetalogprobforage(age,
                                                                                                 beta_prior, beta_for, beta_con, beta_mor)
                                            # see if it's better than top_n; if so, add; if not, ignore
                                            # check size of dataframe
                                            top_n_size = len(age_top_n.index)
                                            if (top_n_size < self.top_n):
                                                # add it
                                                age_top_n.loc[top_n_size] = [sigma_prior, sigma_for, sigma_con, sigma_mor,
                                                                             beta_prior, beta_for, beta_con, beta_mor, this_logprobforage]
                                            else:
                                                # see if the log prob is better than the ones already there
                                                worst_logprob = age_top_n.at[top_n_size - 1, 'logprob']
                                                if this_logprobforage > worst_logprob:
                                                    print("debug, found better sigmas (prior = {}, for = {}, con = {}, mor = {})".format(sigma_prior, sigma_for, sigma_con, sigma_mor))
                                                    print(" and better betas (prior = {}, for = {}, con = {}, mor = {})".format(beta_prior, beta_for, beta_con, beta_mor))
                                                    print("  with logprob {}".format(this_logprobforage))
                                                    age_top_n.drop([top_n_size-1])
                                                    age_top_n.loc[top_n_size-1] = [sigma_prior, sigma_for, sigma_con, sigma_mor,
                                                                                   beta_prior, beta_for, beta_con, beta_mor, this_logprobforage]
                                             # resort on logprob
                                            age_top_n.sort_values(by=['logprob'], inplace=True, ignore_index=True, ascending=False)

        return age_top_n

    # this is a numpy array of subject response probs, with each item having a row (in order from obsexptprobs)
    # each column in order by betas from paper
    # columns: useall,
    #          not(prior), not(form), not(form, prior),
    #          not(con), not(con, prior),
    #          not(mor), not(mor, prior)
    #          not(form, con), not(form, con, prior)
    #          not(con, mor), not(con, mor, prior)
    #          not(form, mor), not(form, mor, prior)
    #          not(form, con, mor), not(form, con, mor, prior)
    # can use it to create rapid weighted logprobs for each item
    def initbetaitemsubjprobsforage(self, age, priors, likelihoods, obsexptprobs):
        # DONE: create numpy array with one row per item with all these normalized subject response probs
        #        already calculated for these priors and likelihoods
        #        Non-subject response probs are then just 1 - these subject response probs
        # and return it
        # create empty numpy array to hold this, with item_number columns and 16 rows
        # first, get the rows of item info, and use the count of those
        age_obsexptresp_df = obsexptprobs.getagerows(age)
        self.num_items = age_obsexptresp_df.shape[0] # good to have number of items in general
        self.betaitemsubjprobsforage = np.zeros((self.num_items, 16))
        # also initialize np array with counts for subj and nonsubj responses
        self.numitemrespsforage = np.zeros((self.num_items, 2))
        # then, get normalized subj probs for this item and put it in the right columns
        row_count = 0;
        for index, row in age_obsexptresp_df.iterrows():

            item_for, item_con, item_mor = row['form'], row['connective'], row['morphology']
            item_subj_mor, item_nons_mor = row['subj_morph'], row['nonsubj_morph']
            item_subj_resp, item_nons_resp = row['num_subj_resp'], row['num_nonsubj_resp']

            index = row_count # so we create this not with the actual index but with the number of items for that age

            # populate numitemrespsforage
            self.numitemrespsforage[index,0] = item_subj_resp
            self.numitemrespsforage[index,1] = item_nons_resp

            # now get column values
            # useall = 0
            norm_prob = self.getnormalizedsubjprob(item_subj_mor, item_nons_mor,
                                                   item_for, item_con, item_mor,
                                                   priors, likelihoods,
                                                   True, True, True, True)
            self.betaitemsubjprobsforage[index,0] = norm_prob
            #          not(prior) = 1, not(form) = 2, not(form, prior) = 3,
            norm_prob = self.getnormalizedsubjprob(item_subj_mor, item_nons_mor,
                                                   item_for, item_con, item_mor,
                                                   priors, likelihoods,
                                                   False, True, True, True)
            self.betaitemsubjprobsforage[index,1] = norm_prob
            norm_prob = self.getnormalizedsubjprob(item_subj_mor, item_nons_mor,
                                                   item_for, item_con, item_mor,
                                                   priors, likelihoods,
                                                   True, False, True, True)
            self.betaitemsubjprobsforage[index,2] = norm_prob
            norm_prob = self.getnormalizedsubjprob(item_subj_mor, item_nons_mor,
                                                   item_for, item_con, item_mor,
                                                   priors, likelihoods,
                                                   False, False, True, True)
            self.betaitemsubjprobsforage[index,3] = norm_prob
            #          not(con) = 4, not(con, prior) = 5,
            norm_prob = self.getnormalizedsubjprob(item_subj_mor, item_nons_mor,
                                                   item_for, item_con, item_mor,
                                                   priors, likelihoods,
                                                   True, True, False, True)
            self.betaitemsubjprobsforage[index,4] = norm_prob
            norm_prob = self.getnormalizedsubjprob(item_subj_mor, item_nons_mor,
                                                   item_for, item_con, item_mor,
                                                   priors, likelihoods,
                                                   False, True, False, True)
            self.betaitemsubjprobsforage[index,5] = norm_prob
            #          not(mor) = 6, not(mor, prior) = 7
            norm_prob = self.getnormalizedsubjprob(item_subj_mor, item_nons_mor,
                                                   item_for, item_con, item_mor,
                                                   priors, likelihoods,
                                                   True, True, True, False)
            self.betaitemsubjprobsforage[index,6] = norm_prob
            norm_prob = self.getnormalizedsubjprob(item_subj_mor, item_nons_mor,
                                                   item_for, item_con, item_mor,
                                                   priors, likelihoods,
                                                   False, True, True, False)
            self.betaitemsubjprobsforage[index,7] = norm_prob
            #          not(form, con) = 8, not(form, con, prior) = 9
            norm_prob = self.getnormalizedsubjprob(item_subj_mor, item_nons_mor,
                                                   item_for, item_con, item_mor,
                                                   priors, likelihoods,
                                                   True, False, False, True)
            self.betaitemsubjprobsforage[index,8] = norm_prob
            norm_prob = self.getnormalizedsubjprob(item_subj_mor, item_nons_mor,
                                                   item_for, item_con, item_mor,
                                                   priors, likelihoods,
                                                   False, False, False, True)
            self.betaitemsubjprobsforage[index,9] = norm_prob
            #          not(con, mor) = 10, not(con, mor, prior) = 11
            norm_prob = self.getnormalizedsubjprob(item_subj_mor, item_nons_mor,
                                                   item_for, item_con, item_mor,
                                                   priors, likelihoods,
                                                   True, True, False, False)
            self.betaitemsubjprobsforage[index,10] = norm_prob
            norm_prob = self.getnormalizedsubjprob(item_subj_mor, item_nons_mor,
                                                   item_for, item_con, item_mor,
                                                   priors, likelihoods,
                                                   False, True, False, False)
            self.betaitemsubjprobsforage[index,11] = norm_prob
            #          not(form, mor) = 12, not(form, mor, prior) = 13
            norm_prob = self.getnormalizedsubjprob(item_subj_mor, item_nons_mor,
                                                   item_for, item_con, item_mor,
                                                   priors, likelihoods,
                                                   True, False, True, False)
            self.betaitemsubjprobsforage[index,12] = norm_prob
            norm_prob = self.getnormalizedsubjprob(item_subj_mor, item_nons_mor,
                                                   item_for, item_con, item_mor,
                                                   priors, likelihoods,
                                                   False, False, True, False)
            self.betaitemsubjprobsforage[index,13] = norm_prob
            #          not(form, con, mor) = 14, not(form, con, mor, prior) = 15
            norm_prob = self.getnormalizedsubjprob(item_subj_mor, item_nons_mor,
                                                   item_for, item_con, item_mor,
                                                   priors, likelihoods,
                                                   True, False, False, False)
            self.betaitemsubjprobsforage[index,14] = norm_prob
            norm_prob = self.getnormalizedsubjprob(item_subj_mor, item_nons_mor,
                                                   item_for, item_con, item_mor,
                                                   priors, likelihoods,
                                                   False, False, False, False)
            self.betaitemsubjprobsforage[index,15] = norm_prob

            row_count += 1

    # return normalized probability for this item's values
    def getnormalizedsubjprob(self, item_subj_mor, item_nons_mor,
                          item_for, item_con, item_mor,
                          priors, likelihoods,
                          use_prior=True, use_for=True, use_con=True, use_mor=True):
        prob_S = self.calculateprob("Subj", item_subj_mor,
                                    item_for, item_con, item_mor,
                                    priors, likelihoods,
                                    use_prior, use_for, use_con, use_mor)
        prob_NS = self.calculateprob("Non", item_nons_mor,
                                    item_for, item_con, item_mor,
                                    priors, likelihoods,
                                    use_prior, use_for, use_con, use_mor)
        # Normalize these
        denom = prob_S + prob_NS
        prob_S = prob_S/denom
        prob_NS = prob_NS/denom
        return prob_S

    # calculates logprobforage with betas for inaccurate deployment
    def calculatebetalogprobforage(self, age,
                                   #priors, likelihoods, obsexptprobs,
                                   #betalogprobs_df,
                                   beta_prior, beta_for, beta_con, beta_mor):
        mixturelogprob = 0.0 # initialize to 0
        # DONE:
        # logprob for age is sum of logprobs for items
        # (1) take self.betaitemsubjprobsforage & create weighted sum of each item's row
        # based on beta_prior, beta_for, beta_con, beta_mor
        # (2) then, use self.numitemrespsforage to create logprob for item
        # sum these to get logprob for age

        # use function to get item's beta prob (weighted_sum) for both subject and non_subject responses
        (norm_weighted_sum_subj, norm_weighted_sum_nonsubj) = self.getnormweightedsum(beta_prior, beta_for, beta_con, beta_mor)

        # now use these to create logprobs for all items, based on self.numitemrespsforage
        # need: for each item: log(prob_S^num_s * prob_NS^num_nons) = num_s*log(prob_S) + num_nons * log(prob_NS)
        #       then can sum logs across all items (since this is the same as multiplying in non-log space)
        logprob_items = np.zeros(self.num_items)
        logprob_items = self.numitemrespsforage[:,0]*np.log(norm_weighted_sum_subj) + self.numitemrespsforage[:,1]*np.log(norm_weighted_sum_nonsubj)
        mixturelogprob = logprob_items.sum()

        return mixturelogprob

    def getnormweightedsum(self, beta_prior, beta_for, beta_con, beta_mor):

        # create np array of weighted sum of each item's row, so need one per item row
        betaweights_row = self.initbetaweights(beta_prior, beta_for, beta_con, beta_mor)
        # make one row of this per item for fast multiplication
        betaweights = np.zeros((self.num_items, 16))
        for item in range(self.num_items):
            betaweights[item] = betaweights_row

        # subj
        weighted_sum_subj = self.getweightedsum("subj", betaweights, beta_prior, beta_for, beta_con, beta_mor)
        # non-subj
        weighted_sum_nonsubj = self.getweightedsum("nonsubj", betaweights, beta_prior, beta_for, beta_con, beta_mor)

        # Normalize weighted_sum_subj & nonsubj
        norm_weighted_sum_subj = np.zeros(self.num_items)
        norm_weighted_sum_subj = weighted_sum_subj/(weighted_sum_subj + weighted_sum_nonsubj)
        norm_weighted_sum_nonsubj = np.zeros(self.num_items)
        norm_weighted_sum_nonsubj = weighted_sum_nonsubj/(weighted_sum_subj + weighted_sum_nonsubj)

        return (norm_weighted_sum_subj, norm_weighted_sum_nonsubj)

    def getweightedsum(self, subjornon, betaweights,
                       beta_prior, beta_for, beta_con, beta_mor):
        weighted_sum = np.zeros(self.num_items)

        # get individual weighted values
        weighted_indiv_probs = np.zeros((self.num_items, 16))
        if(subjornon == "subj"):
            weighted_indiv_probs = self.betaitemsubjprobsforage * betaweights
        else: # nonsubj
            weighted_indiv_probs = (1 - self.betaitemsubjprobsforage) * betaweights
        # get sum of these weighted probs for each item
        for item in range(self.num_items):
            weighted_sum[item] = weighted_indiv_probs[item].sum()

        return weighted_sum

    def initbetaweights(self, beta_prior, beta_for, beta_con, beta_mor):
        # returns np.array with one row containing the 16 beta weights
        betaweights = np.zeros(16)
        # useall = 0
        # beta_for * beta_con * beta_mor * beta_prior * thislogprob
        betaweights[0] = beta_for * beta_con * beta_mor * beta_prior
        # not(prior)
        # beta_for * beta_con * beta_mor * (1-beta_prior)
        betaweights[1] = beta_for * beta_con * beta_mor * (1-beta_prior)
        # not(for)
        # (1-beta_for) * beta_con * beta_mor * beta_prior
        betaweights[2] = (1-beta_for) * beta_con * beta_mor * beta_prior
        #  not(for, prior)
        #  (1-beta_for) * beta_con * beta_mor * (1-beta_prior)
        betaweights[3] = (1-beta_for) * beta_con * beta_mor * (1-beta_prior)
        # not(con)
        # beta_for * (1-beta_con) * beta_mor * beta_prior
        betaweights[4] = (1-beta_con) * beta_mor * beta_prior
        # not(con, prior)
        # beta_for * (1-beta_con) * beta_mor * (1-beta_prior)
        betaweights[5] = beta_for * (1-beta_con) * beta_mor * (1-beta_prior)
        # not(mor)
        # beta_for * beta_con * (1-beta_mor) * beta_prior
        betaweights[6] = beta_for * beta_con * (1-beta_mor) * beta_prior
        # not(mor, prior)
        #  beta_for * beta_con * (1-beta_mor) * (1-beta_prior)
        betaweights[7] = beta_for * beta_con * (1-beta_mor) * (1-beta_prior)
        # not(for, con)
        # (1-beta_for) * (1-beta_con) * beta_mor * beta_prior
        betaweights[8] = (1-beta_for) * (1-beta_con) * beta_mor * beta_prior
        # not(for, con, prior)
        #  (1-beta_for) * (1-beta_con) * beta_mor * (1-beta_prior)
        betaweights[9] = (1-beta_for) * (1-beta_con) * beta_mor * (1-beta_prior)
        # not(con, mor)
        # beta_for * (1-beta_con) * (1-beta_mor) * beta_prior
        betaweights[10] = beta_for * (1-beta_con) * (1-beta_mor) * beta_prior
        # not(con, mor, prior)
        # beta_for * (1-beta_con) * (1-beta_mor) * (1-beta_prior)
        betaweights[11] = beta_for * (1-beta_con) * (1-beta_mor) * (1-beta_prior)
        # not(for, mor)
        # (1-beta_for) * beta_con * (1-beta_mor) * beta_prior
        betaweights[12] = (1-beta_for) * beta_con * (1-beta_mor) * beta_prior
        # not(for, mor, prior)
        #  (1-beta_for) * beta_con * (1-beta_mor) * (1-beta_prior)
        betaweights[13] = (1-beta_for) * beta_con * (1-beta_mor) * (1-beta_prior)
        # not(for, con, mor)
        # (1-beta_for) * (1-beta_con) * (1-beta_mor) * beta_prior
        betaweights[14] = (1-beta_for) * (1-beta_con) * (1-beta_mor) * beta_prior
        # not(for, con, mor, prior)
        # (1-beta_for) * (1-beta_con) * (1-beta_mor) * (1-beta_prior)
        betaweights[15] = (1-beta_for) * (1-beta_con) * (1-beta_mor) * (1-beta_prior)

        return betaweights

    def getbetalogprobfromdf(self, betalogprobs_df, use_prior, use_for, use_con, use_mor):
        thisrow_df = betalogprobs_df.loc[(betalogprobs_df['use_for'] == use_for) & (betalogprobs_df['use_con'] == use_con) & \
                                       (betalogprobs_df['use_mor'] == use_mor) & (betalogprobs_df['use_prior'] == use_prior)]
        thislogprob = thisrow_df.iloc[0]['logprob']
        return thislogprob

    # calculates BIC for age's obs_expt_items, given age, log prob, obsexptprobs object, and free parameter num m
    def calculateBICforage(self, age, agelogprob, obsexptprobs, m):
        # BIC = m * log(size(data)) - 2 * agelogprob
        size_data = obsexptprobs.getagesize(age)
        print("debug, data size for BIC is {}".format(size_data))
        # then calculate BIC = m * log(size_data) - 2*logprob
        BIC = m * math.log(size_data) - 2 * agelogprob
        return BIC

    # calculates log prob for age's obs expt items, given priors, likelihoods, and obsexptprobs object
    # def calculatelogprobforage(self, age, corpusprobs, obsexptprobs):
    def calculatelogprobforage(self, age, priors, likelihoods, obsexptprobs,
                               use_for=True,  # use booleans, default = true
                               use_con=True,
                               use_mor=True,
                               use_prior=True
                               ):
        # Use corpusprobs and obsexptprobs data frame to calculate log prob for all stimuli for that age
        # need: for each item: log(prob_S^num_s * prob_NS^num_nons) = num_s*log(prob_S) + num_nons * log(prob_NS)
        #       then can sum logs across all items (since this is the same as multiplying in non-log space)
        # first, get data rows corresponding to age
        age_obsexptresp_df = obsexptprobs.getagerows(age)
        #print("debug: age {} obsexptresp rows has {} rows".format(age, age_obsexptresp_df.shape))
        #age_obsexptresp_df.to_csv(path_or_buf=sys.stdout, sep=' ', index=False)
        # for each expt item (and take log sum across all item probabilities):
        #        get feature values, and subj and non-subj morphology
        #        then can pull appropriate priors and likelihoods
        #        then can do appropriate posterior calculation
        # initialize logprobtotal to 0
        logprobtotal = 0
        for index, row in age_obsexptresp_df.iterrows():
            item_for, item_con, item_mor = row['form'], row['connective'], row['morphology']
            item_subj_mor, item_nons_mor = row['subj_morph'], row['nonsubj_morph']
            item_subj_resp, item_nons_resp = row['num_subj_resp'], row['num_nonsubj_resp']
            ## this is the part that will change by model variant type -- except not really
            # for all model types
            # 1 (baseline) use true corpusprobs, calculate posterior without mixture
            # 2 (inacc rep) use sigma-ed corpusprobs, calculate posterior without mixture
            # 3 (inacc deploy) called as part of mixture model
            # 4 (both inacc) use sigma-ed corpusprobs, called as part of mixture model
            # for each stimulus in age, calculate logprob, using all feature values and prior
            logprobitem = self.calculatelogprobforitem(item_for, item_con, item_mor,
                                            item_subj_mor, item_nons_mor,
                                            item_subj_resp, item_nons_resp,
                                            priors,
                                            likelihoods,
                                            use_for,
                                            use_con,
                                            use_mor,
                                            use_prior
                                            )
            logprobtotal = logprobtotal + logprobitem

        return logprobtotal

    # calculate log probs for specific experimental item with specific priors and likelihoods
    # with certain feature values in use (to handle inaccurate deployment)
    def calculatelogprobforitem(self, item_for, item_con, item_mor,
                                item_subj_mor, item_nons_mor,
                                item_subj_resp, item_nons_resp,
                                priors, # should be hash
                                likelihoods, # should be hash
                                use_for = True, # use booleans, default = true
                                use_con = True,
                                use_mor = True,
                                use_prior = True
                                ):
        # log(prob_S^num_s * prob_NS^num_nons) = num_s*log(prob_S) + num_nons * log(prob_NS)
        # first calculate the probability of the subject responses (prob_S)
        # then the probability of the non-subject responses (prob_NS)

        prob_S = self.calculateprob("Subj", item_subj_mor,
                                    item_for, item_con, item_mor,
                                    priors, likelihoods,
                                    use_prior, use_for, use_con, use_mor)
        prob_NS = self.calculateprob("Non", item_nons_mor,
                                     item_for, item_con, item_mor,
                                     priors, likelihoods,
                                     use_prior, use_for, use_con, use_mor)
        # Normalize these
        denom = prob_S + prob_NS
        prob_S = prob_S/denom
        prob_NS = prob_NS/denom

        # log(prob_S^num_s * prob_NS^num_nons) = num_s*log(prob_S) + num_nons * log(prob_NS)
        # logprob for item = item_subj_resp * log(prob_S) + item_nons_resp * log(prob_NS)
        logprob = item_subj_resp * math.log(prob_S) + item_nons_resp * math.log(prob_NS)
        return logprob

    def calculateprob(self, subjornon, item_ante_mor,
                      item_for, item_con, item_mor,
                      priors, likelihoods,
                      use_prior, use_for, use_con, use_mor):
        # probability of subject responses prob_S
        prob = 1 # initialize prob to 1, will multiple with other things
        if use_prior:
            # get appropriate prior value
            # name = "ante" + "Subj/Non" + Sg/Pl (so convert subj_mor to initial caps)
            prior_string = "ante" + subjornon + item_ante_mor.capitalize()
            prob = priors[prior_string]
        if use_for:
            # get appropriate form likelihood value for item_for, given subject/non is item_ante_mor
            # name = Null/Overt + "Subj/Non" + "Sg/Pl"
            form_string = item_for.capitalize() + subjornon + item_ante_mor.capitalize()
            prob = prob * likelihoods[form_string]
        if use_con:
            # get appropriate form likelihood value for item_con, given subject is item_ante_mor
            # name = Desp/Porq + "Subj/Non" + "Sg/Pl"
            con_string = item_con[0:4].capitalize() + subjornon + item_ante_mor.capitalize()
            prob = prob * likelihoods[con_string]
        if use_mor:
            # get appropriate form likelihood value for item_mor, given subject is item_ante_mor
            # name = "Sg/Pl" + "Subj" + "Sg/Pl"
            mor_string = item_mor.capitalize() + subjornon + item_ante_mor.capitalize()
            prob = prob * likelihoods[mor_string]
        return prob

    def print_logprobandBIC(self, output_file=sys.stdout, precision=5):
        output_file.write("Model type {}: ".format(self.model_type))
        if self.model_type == 1:
            output_file.write("Baseline model, m = {}\n".format(self.m))
            # model variables: self.age_XX_logprob, self.age_XX_BIC
            if (not self.only_one_age) or ((self.only_one_age) and (self.which_age == "3")):
                output_file.write("age 3 logprob = {}, age 3 BIC = {}\n".format(self.age_3_logprob, self.age_3_BIC))
            if (not self.only_one_age) or ((self.only_one_age) and (self.which_age == "4")):
                output_file.write("age 4 logprob = {}, age 4 BIC = {}\n".format(self.age_4_logprob, self.age_4_BIC))
            if (not self.only_one_age) or ((self.only_one_age) and (self.which_age == "5")):
                output_file.write("age 5 logprob = {}, age 5 BIC = {}\n".format(self.age_5_logprob, self.age_5_BIC))
            if (not self.only_one_age) or ((self.only_one_age) and (self.which_age == "adult")):
                output_file.write("age adult logprob = {}, age adult BIC = {}\n".format(self.age_adult_logprob, self.age_adult_BIC))
        else:
            has_sigmas = False
            has_betas = False
            if self.model_type == 2:
                output_file.write("Inaccurate representations, m = {}\n".format(self.m))
                has_sigmas = True
            elif self.model_type == 3:
                output_file.write("Inaccurate deployment, m = {}\n".format(self.m))
                has_betas = True
            else:
                output_file.write("Both inaccurate, m = {}\n".format(self.m))
                has_sigmas = True
                has_betas = True
            # print them out with decimals rounded to precision (default 5)
            if (not self.only_one_age) or ((self.only_one_age) and (self.which_age == "3")):
                output_file.write("age 3 top {} logprob and BICs\n".format(self.top_n))
                self.age_3_top_n.round(precision).to_csv(output_file, sep="\t")
                self.printagerange(self.age_3_top_n, has_sigmas, has_betas, output_file)
            if (not self.only_one_age) or ((self.only_one_age) and (self.which_age == "4")):
                output_file.write("age 4 top {} logprob and BICs\n".format(self.top_n))
                self.age_4_top_n.round(precision).to_csv(output_file, sep="\t")
                self.printagerange(self.age_4_top_n, has_sigmas, has_betas, output_file)
            if (not self.only_one_age) or ((self.only_one_age) and (self.which_age == "5")):
                output_file.write("age 5 top {} logprob and BICs\n".format(self.top_n))
                self.age_5_top_n.round(precision).to_csv(output_file, sep="\t")
                self.printagerange(self.age_5_top_n, has_sigmas, has_betas, output_file)
            if (not self.only_one_age) or ((self.only_one_age) and (self.which_age == "adult")):
                output_file.write("age adult top {} logprob and BICs\n".format(self.top_n))
                self.age_adult_top_n.round(precision).to_csv(output_file, sep="\t")
                self.printagerange(self.age_adult_top_n, has_sigmas, has_betas, output_file)

    def printagerange(self, topn_df, has_sigmas, has_betas, output_file):
        if (has_sigmas):
            self.printrange(topn_df, "sigma", "prior", output_file)
            self.printrange(topn_df, "sigma", "for", output_file)
            self.printrange(topn_df, "sigma", "con", output_file)
            self.printrange(topn_df, "sigma", "mor", output_file)
        if (has_betas):
            self.printrange(topn_df, "beta", "prior", output_file)
            self.printrange(topn_df, "beta", "for", output_file)
            self.printrange(topn_df, "beta", "con", output_file)
            self.printrange(topn_df, "beta", "mor", output_file)
        output_file.write("\n")

    def printrange(self, topn_df, feature_type, feature, output_file):
        feature_string = feature_type + "_" + feature
        min_feat_type = topn_df[feature_string].min()
        max_feat_type = topn_df[feature_string].max()
        output_file.write("\t{}: {}-{}\n".format(feature_string, min_feat_type, max_feat_type))

    def print_modelsettings(self, output_file=sys.stdout):
        output_file.write("Model type {}: ".format(self.model_type))
        if self.model_type == 1:
            output_file.write("Baseline model\n")
        elif self.model_type == 2:
            output_file.write("Inaccurate representations\n")
        elif self.model_type == 3:
            output_file.write("Inaccurate deployment\n")
        else:
            output_file.write("Both inaccurate\n")
        if (self.model_type == 2) or (self.model_type == 4):
            output_file.write("{} <= sigma_prior <= {} by {}\n".format(self.sigma_prior_low, self.sigma_prior_high, self.sigma_prior_inc))
            output_file.write("{} <= sigma_form <= {} by {}\n".format(self.sigma_form_low, self.sigma_form_high, self.sigma_form_inc))
            output_file.write("{} <= sigma_con <= {} by {}\n".format(self.sigma_con_low, self.sigma_con_high, self.sigma_con_inc))
            output_file.write("{} <= sigma_mor <= {} by {}\n".format(self.sigma_mor_low, self.sigma_mor_high, self.sigma_mor_inc))
        if (self.model_type == 3) or (self.model_type == 4):
            output_file.write("{} <= beta_prior <= {} by {}\n".format(self.beta_prior_low, self.beta_prior_high, self.beta_prior_inc))
            output_file.write("{} <= beta_form <= {} by {}\n".format(self.beta_form_low, self.beta_form_high, self.beta_form_inc))
            output_file.write("{} <= beta_con <= {} by {}\n".format(self.beta_con_low, self.beta_con_high, self.beta_con_inc))
            output_file.write("{} <= beta_mor <= {} by {}\n".format(self.beta_mor_low, self.beta_mor_high, self.beta_mor_inc))
        output_file.write("Getting top {} options\n".format(self.top_n))
        output_file.write("M for BIC calculation is {}\n".format(self.m))
        if(self.only_one_age):
            output_file.write("Only doing age {}\n".format(self.which_age))

######################
class ObservedExptResponses:
    # initialize observed experiment responses from input exptobs_file
    def __init__(self,exptobsfile):
        print('Reading observed experiment responses from {}'.format(exptobsfile))
        # use a dataframe for this
        # data has the form below, with # comments interspersed:
        # age,connective,form,morphology,subj_mor,nonsubj_mor,num_subj_resp,num_nonsubj_resp
        # age 3
        # 3,despues,null,sg,sg,pl,14,22
        self.obsexptresp_df = pd.read_csv(exptobsfile, keep_default_na=False)
        # convert various rows to numeric values: num_subj_resp, num_nonsubj_resp
        # (note: age is mixed value, so keep as string)
        self.obsexptresp_df["num_subj_resp"] = pd.to_numeric(self.obsexptresp_df["num_subj_resp"])
        self.obsexptresp_df["num_nonsubj_resp"] = pd.to_numeric(self.obsexptresp_df["num_nonsubj_resp"])

    def getagerows(self, age):
        # convert age to string, since could be "adult"
        return self.obsexptresp_df[self.obsexptresp_df["age"] == str(age)]

    def getagesize(self, age):
    #     # sum over num_subj_resp and num_nonsubj_resp for reach row in age
        agerows = self.getagerows(age)
        return agerows['num_subj_resp'].sum() + agerows['num_nonsubj_resp'].sum()

    def print_obsexptresp(self, output_file=sys.stdout):
        self.obsexptresp_df.to_csv(path_or_buf=output_file, sep=' ', index=False)



#######################
class CorpusProbabilities:
    # initialize corpus probabilities from input corpusprob_file
    def __init__(self, corpusfile):
        # make hash where key = name of prob and value = prob
        # one hash for priors and one hash for likelihoods
        # priors[ante-antesgvspl-antesubjvsnot] = value
        # likelihoods[form/con/mor val-antesgvspl-antesubjvsnot] = value
        self.priors = {} # initialize priors hash to be empty
        self.likelihoods = {} # initialize likelihoods hash to be empty

        # open corpusprob_file for reading in
        print('Reading corpus probabilities from {}'.format(corpusfile))
        with open(corpusfile,'r') as infile:
        #### of this format: ####
        # priors from corpus data, S = Subject, NS = Non-Subject
        # anteSSg 0.3622047  # a=Subj Sg
        # ...(3 more rows)
        # # likelihoods from corpus data, arranged to follow order in table in paper
        # # first row: Subj Sg (form, connective, morphology)
        # Null_giv_anteSSg 0.9378882 # a=Subj Sg, for=null|a
        # ...
            inline = infile.readline()
            while inline:
                # ignore any lines that begin with # since those are comments
                pattern = re.compile('^\#')
                match = pattern.search(inline) # is that what inline has?
                if not match:
                    # continue on
                    # priors have this form
                    # anteSSg 0.3622047  # a=Subj Sg
                    # is it a prior? if so, populate priors hash
                    pattern = re.compile('^ante(N?S)(\w\w) (\d\.\d+) ')
                    match = pattern.search(inline)
                    if match:
                        if match.group(1) == 'S':
                            antesubjnon = 'Subj'
                        elif match.group(1) == 'NS':
                            antesubjnon = 'Non'
                        # match.group(2): Sg or Pl
                        label = "ante" + antesubjnon + match.group(2)
                        self.priors[label] = float(match.group(3))
                    else:
                    # else: likelihoods have this form
                    # Null_giv_anteSSg 0.9378882 # a=Subj Sg, for=null|a
                    # is it a likelihood? if so, population likelihoods hash
                        pattern = re.compile('^(\w+)\_giv\_ante(N?S)(\w\w) (\d\.\d+) ')
                        match = pattern.search(inline)
                        if match:
                            # want label = featureval-antesubjvsnot-antesgvspl
                            # match.group(1) is the feature value: Null, Overt, Desp, Porq, Sg, Pl
                            if match.group(2) == 'S':
                                antesubjnon = 'Subj'
                            elif match.group(2) == 'NS':
                                antesubjnon = 'Non'

                            # match.group(3) = Sg or Pl
                            label = match.group(1) + antesubjnon + match.group(3)
                            self.likelihoods[label] = float(match.group(4))

                # read in next inline
                inline = infile.readline()

    def get_sigmalikelihoods(self, sigma_for, sigma_con, sigma_mor):
        # need to get likelihoods after they're been raised to appropriate sigmas and renormalized (in pairs)
        # sigmas hash in order: DespNonPl, DespNonSg, DespSubjPl, DespSubjSg
        #                              Null...,
        #                              Overt...,
        #                              Pl...,
        #                              Porq...,
        #                              Sg....
        sigma_likelihoods_hash = {} # initially empty
        # start with sigma_for (Null vs Overt): go through SubjSg , SubjPl, NonSg, NonPl
        sigma_likelihoods_hash = self.raiserenorm("Null", "Overt", "Subj", "Sg", sigma_for, sigma_likelihoods_hash)
        sigma_likelihoods_hash = self.raiserenorm("Null", "Overt", "Subj", "Pl", sigma_for, sigma_likelihoods_hash)
        sigma_likelihoods_hash = self.raiserenorm("Null", "Overt", "Non", "Sg", sigma_for, sigma_likelihoods_hash)
        sigma_likelihoods_hash = self.raiserenorm("Null", "Overt", "Non", "Pl", sigma_for, sigma_likelihoods_hash)

        # now sigma_con (Desp vs Porq): go through SubjSg , SubjPl, NonSg, NonPl
        sigma_likelihoods_hash = self.raiserenorm("Desp", "Porq", "Subj", "Sg", sigma_con, sigma_likelihoods_hash)
        sigma_likelihoods_hash = self.raiserenorm("Desp", "Porq", "Subj", "Pl", sigma_con, sigma_likelihoods_hash)
        sigma_likelihoods_hash = self.raiserenorm("Desp", "Porq", "Non", "Sg", sigma_con, sigma_likelihoods_hash)
        sigma_likelihoods_hash = self.raiserenorm("Desp", "Porq", "Non", "Pl", sigma_con, sigma_likelihoods_hash)

        # now sigma_mor (Sg vs Pl): go through SubjSg , SubjPl, NonSg, NonPl
        sigma_likelihoods_hash = self.raiserenorm("Sg", "Pl", "Subj", "Sg", sigma_mor, sigma_likelihoods_hash)
        sigma_likelihoods_hash = self.raiserenorm("Sg", "Pl", "Subj", "Pl", sigma_mor, sigma_likelihoods_hash)
        sigma_likelihoods_hash = self.raiserenorm("Sg", "Pl", "Non", "Sg", sigma_mor, sigma_likelihoods_hash)
        sigma_likelihoods_hash = self.raiserenorm("Sg", "Pl", "Non", "Pl", sigma_mor, sigma_likelihoods_hash)

        return sigma_likelihoods_hash

    def raiserenorm(self, featval1, featval2, subjvsnon, sgvspl, sigma, sigma_likelihoods_hash):
        # create numpy array with the two original values from self.likelihoods
        label1 = featval1 + subjvsnon + sgvspl
        label2 = featval2 + subjvsnon + sgvspl
        likelihoods_pair = np.array([self.likelihoods[label1], self.likelihoods[label2]])
        # raise to sigma
        likelihoods_pair = np.power(likelihoods_pair, sigma)
        # renorm
        norm_likelihoods = np.sum(likelihoods_pair)
        normed_likelihoods_pair = likelihoods_pair/norm_likelihoods
        # put the two values in the sigma_likelihoods_hash
        sigma_likelihoods_hash[label1] = normed_likelihoods_pair[0]
        sigma_likelihoods_hash[label2] = normed_likelihoods_pair[1]
        return sigma_likelihoods_hash

    def get_sigmapriors(self, sigma_prior):
        # need to get priors after they've been raised to sigma_prior and renormalized
        # create numpy array in order anteNonPl, anteNonSg, anteSubjPl, anteSubjSg
        orig_priors = np.array([self.priors['anteNonPl'], self.priors['anteNonSg'],
                                 self.priors['anteSubjPl'], self.priors['anteSubjSg']])
        sigma_priors = np.power(orig_priors, sigma_prior)
        # now normalize
        norm_priors = np.sum(sigma_priors)
        normed_sigma_priors = sigma_priors/norm_priors
        # populate sigma priors hash
        sigma_priors_hash = {}
        sigma_priors_hash['anteNonPl'] = normed_sigma_priors[0]
        sigma_priors_hash['anteNonSg'] = normed_sigma_priors[1]
        sigma_priors_hash['anteSubjPl'] = normed_sigma_priors[2]
        sigma_priors_hash['anteSubjSg'] = normed_sigma_priors[3]
        return sigma_priors_hash

    def get_priors(self):
        return self.priors

    def get_likelihoods(self):
        return self.likelihoods

    # function to print out priors and likelihoods from corpus
    def print_corpus_info(self, output_file=sys.stdout):
        # print priors
        output_file.write("Corpus prior probabilities:\n")
        print_hash(self.priors, output_file)
        # print likelihoods
        output_file.write("Corpus likelihood probabilities:\n")
        print_hash(self.likelihoods, output_file)

    # any future functions to access priors or likelihoods on CorpusProbabilities

##################





# function to read in arguments from the command line
def read_arguments():
    # required:
    #  input file with corpus probabilities (ex: input/corpusprobabilities.txt)
    #  input file with experimental observations (ex: input/exptobserved.txt)
    # optional:
    #  output file to print our results (default: pronounmodels.out)
    #  model type to calculate (1=baseline [default], 2=inacc rep, 3=inacc deploy, 4=inacc both)
    #  sigma range for inacc rep & inacc both (default 0.01 <= sigma <= 4.00, by increment 0.01)
    #  beta range for inacc deploy & inacc both (default 0.00 <= beta <= 1.00, by increment 0.01)
    #  top N parameter sets stored for inacc variants (default 10)
    parser = argparse.ArgumentParser(description="Pronoun interpretation models of observed experimental data")
    parser.add_argument("corpusprob_file", type=str,
                        help="input file containing corpus probabilities") # required argument
    parser.add_argument("exptobs_file", type=str,
                        help="input file containing observed experiment responses") # required argument
    parser.add_argument("--output_file", "-o", type=str, default="pronounmodels.out",
                        help="output file name")
    parser.add_argument("--model_type", "-mt", type=int, default=1,
                        help="model type 1=baseline 2=inaccurate representations 3=inaccurate deployment 4=inaccurate both")
    # parameter settings for sigma, beta
    # prior
    parser.add_argument("--sigma_prior_low", "-spl", type=float, default=0.1,
                        help="lower bound for sigma prior parameter")
    parser.add_argument("--sigma_prior_high", "-sph", type=float, default=4.00,
                        help="upper bound for sigma prior parameter")
    parser.add_argument("--sigma_prior_inc", "-spi", type=float, default=0.1,
                        help="increment for sigma prior parameter")
    parser.add_argument("--beta_prior_low", "-bpl", type=float, default=0.1,
                        help="lower bound for beta prior parameter")
    parser.add_argument("--beta_prior_high", "-bph", type=float, default=1.00,
                        help="upper bound for beta prior parameter")
    parser.add_argument("--beta_prior_inc", "-bpi", type=float, default=0.1,
                        help="increment for beta prior parameter")
    # top N stored for model variants that require parameter search
    # for
    parser.add_argument("--sigma_form_low", "-sfl", type=float, default=0.1,
                        help="lower bound for sigma form parameter")
    parser.add_argument("--sigma_form_high", "-sfh", type=float, default=4.00,
                        help="upper bound for sigma form parameter")
    parser.add_argument("--sigma_form_inc", "-sfi", type=float, default=0.1,
                        help="increment for sigma form parameter")
    parser.add_argument("--beta_form_low", "-bfl", type=float, default=0.1,
                        help="lower bound for beta form parameter")
    parser.add_argument("--beta_form_high", "-bfh", type=float, default=1.00,
                        help="upper bound for beta form parameter")
    parser.add_argument("--beta_form_inc", "-bfi", type=float, default=0.1,
                        help="increment for beta form parameter")
    # con
    parser.add_argument("--sigma_con_low", "-scl", type=float, default=0.1,
                        help="lower bound for sigma con parameter")
    parser.add_argument("--sigma_con_high", "-sch", type=float, default=4.00,
                        help="upper bound for sigma con parameter")
    parser.add_argument("--sigma_con_inc", "-sci", type=float, default=0.1,
                        help="increment for sigma con parameter")
    parser.add_argument("--beta_con_low", "-bcl", type=float, default=0.1,
                        help="lower bound for beta con parameter")
    parser.add_argument("--beta_con_high", "-bch", type=float, default=1.00,
                        help="upper bound for beta con parameter")
    parser.add_argument("--beta_con_inc", "-bci", type=float, default=0.1,
                        help="increment for beta con parameter")
    # mor
    parser.add_argument("--sigma_mor_low", "-sml", type=float, default=0.1,
                        help="lower bound for sigma morphology parameter")
    parser.add_argument("--sigma_mor_high", "-smh", type=float, default=4.00,
                        help="upper bound for sigma morphology parameter")
    parser.add_argument("--sigma_mor_inc", "-smi", type=float, default=0.1,
                        help="increment for sigma morphology parameter")
    parser.add_argument("--beta_mor_low", "-bml", type=float, default=0.1,
                        help="lower bound for beta morphology parameter")
    parser.add_argument("--beta_mor_high", "-bmh", type=float, default=1.00,
                        help="upper bound for beta morphology parameter")
    parser.add_argument("--beta_mor_inc", "-bmi", type=float, default=0.1,
                        help="increment for beta morphology parameter")
    parser.add_argument("--topn", "-tn", type=int, default=10,
                        help="number of top parameter settings stored")
    parser.add_argument("--only_one_age", "-ooa", action="store_true",
                        help="do only for one age specified by age parameter")
    parser.add_argument("--age", "-a", type=str, default="3",
                        help="do only for one age specified by age parameter in string format like \"3\" or \"adult\"")
    return parser.parse_args()

#### helper functions
# generic function for printing out a hash's keys and values in sorted order
def print_hash(to_print_hash, outputfile=sys.stdout):
   for hash_k, hash_v in sorted(to_print_hash.items()):
       #print("{}: {}".format(hash_k, hash_v))
       outputfile.write("{}: {}\n".format(hash_k, hash_v))
       #print("")
       outputfile.write("")

# useful for being able to define functions lower down than where they're called
# e.g., putting main function first and then function definitions below it
if __name__ == "__main__":
    main()