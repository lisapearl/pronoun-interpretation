#### General note, 12/12/25: This is code authored over several iterations by my co-author and then me. 
## It's messy and inefficient. Apologies. 

library(tidyverse)
library(lme4)
library(stringr)
library(plyr)
library(dplyr)
library(reshape2)
library(ggplot2)
library(emmeans)
library(boot)

#Read in aduchi data set (must be located in same folder as this script).
aduchi <- read.table("aduchi-MX1819.txt", header=TRUE)

###########
#exclusions
###########
#take out ger, who accidentally did day 'a' twice (1 adult; experimenter error)
aduchi_clean <- subset(aduchi, participant!="ger")
#take out mxv, who is in speech therapy (na, since mvx < 5 years old)
aduchi_clean <- subset(aduchi_clean, participant!="mxv") 
# remove all kids under 5 for the analyses reported in the 2025 Ms
aduchi_5up <- subset(aduchi_clean, age_y %in% c(5, 6, 'adu'))

#summarize answers vs. errors/non-responses by age group (grouping 6 with 5)
aduchi_5up %>% 
  mutate(response = revalue(response.corr, c("na"="error", "0"="recorded", "1"="recorded"))) %>%
  mutate(age_grp = revalue(age_y, c("6"="5"))) %>%
  select(age_grp, response) %>%
  table()
  
#remove lost answers and non-responses
errors <- subset(aduchi_5up, response.corr %in% c("error", "na"))
aduchi_5up <- subset(aduchi_5up, response.corr!="error")
aduchi_5up <- subset(aduchi_5up, response.corr!="na")

# convert response.corr to a numeric variable
aduchi_5up$response.corr <- as.numeric(paste(aduchi_5up$response.corr))
# convert response.subj into a numeric variable
aduchi_5up$response.subj <- as.numeric(paste(aduchi_5up$response.subj))

#remove practice and filler items
fillers_5up <- subset(aduchi_5up, condition == "filler")
practice_5up <- subset(aduchi_5up, condition == "practice")
aduchi_5up <- subset(aduchi_5up, condition != "practice")
aduchi_5up <- subset(aduchi_5up, condition != "filler")


############################
#participant characteristics
############################

chi_5up <- subset(aduchi_5up, age_y != "adu")
#   Child Ns and age ranges (year; month format) and groups (grouping 6 with 5)
#  for 5 and up: recode 6 as 5+
chi_5up$age_y <- as.numeric(paste(revalue(chi_5up$age_y, c("6"="5"))))
chi_5up$age_m <- as.numeric(paste(chi_5up$age_m))
byindividual_5up <- ddply(chi_5up, .(participant, age_y, sex), summarise, age_m=mean(age_m))
#table of child Ns and age ranges
chiNandR_5up <- ddply(byindividual_5up, .(age_y), summarise, N=n_distinct(participant), 
                  N_f=length(which(sex=="f")),
                  M=round(mean(age_m), 1), Min=min(age_m), Max=max(age_m),
                  M_ym=paste0(mean(age_m)%/%12, ";", round(mean(age_m)%%12)), 
                              Min_ym=paste0(min(age_m)%/%12, ";", min(age_m)%%12), 
                              Max_ym=paste0(max(age_m)%/%12, ";", max(age_m)%%12))


##Adult Ns and "age"
adu_5up <- subset(aduchi_5up, age_y == "adu")
aduN_5up <- n_distinct(adu_5up$participant)
aduN_f5up <- n_distinct(adu_5up$participant[adu_5up$sex=="f"])
  
##Combine the child and adult Ns and age ranges
adurow_5up <- c("adu", aduN_5up, aduN_f5up, "adu", "adu", "adu", "adu", "adu", "adu")
partNandR_5up <- rbind(chiNandR_5up, adurow_5up)
partNandR_5up
  
#total number of all kids together: 
sum(as.numeric(paste(partNandR_5up$N[partNandR_5up$age_y!="adu"])))
#total number of all girls together: 
sum(as.numeric(paste(partNandR_5up$N_f[partNandR_5up$age_y!="adu"])))
  
############################
#division across experiments
############################
#Ns by age group and experiment type
participants_5up <- aduchi_5up %>% select(participant, age_y, sex, type)
participants_5up$age_y <- revalue(participants_5up$age_y, c("6"="5"))
participantNs_5up <- ddply(participants_5up, .(age_y, experiment_type = type), summarise, N=n_distinct(participant))
participantNs_5up

  
############################
#filler scores
############################

#get filler scores by participant
fillscore_5up <- ddply(fillers_5up, .(participant), summarise, filler_score=mean(response.corr))

#add this calculation to the original aduchi dataframe
aduchi_5up <- merge(aduchi_5up, fillscore_5up, by="participant")

#average filler scores by age group
fillscorebyage_5up <- ddply(fillers_5up, .(participant, age_y), summarise, filler_score=mean(response.corr))
fillscorebyage_5up$age_grp <- revalue(fillscorebyage_5up$age_y, c("6"="5"))
meanfillscore_5up <- ddply(fillscorebyage_5up, .(age_grp), summarise, filler_score=mean(filler_score))

#are filler scores above chance in all age groups?
#5-year-olds: yes
t.test(fillscorebyage_5up$filler_score[fillscorebyage_5up$age_grp==5], mu=.5, alternative = "greater")
#adults: yes
t.test(fillscorebyage_5up$filler_score[fillscorebyage_5up$age_grp=="adu"], mu=.5, alternative = "greater") 

## Try a binomial test for the fillers 
#   (basically ignores individual participant contribution to total)
# Need to group the fillers_5up data by participant and perform a binomial test
# adults
fillers_5upadults = subset(fillers_5up, age_y == "adu")
fillers_5upkids = subset(fillers_5up, age_y != "adu")

fillers_5upadults_binomial <- fillers_5upadults %>%
  group_by(participant) %>%
  summarise(
    successes = sum(response.corr),
    trials = length(response.corr)
  ) %>%
  rowwise() %>%
  mutate(
    binom_test = list(binom.test(successes, trials, p=0.5, alternative="greater")),
    p_value = binom_test$p_value
  ) %>%
  ungroup()
print(fillers_5upadults_binomial$binom_test)

fillers_5upkids_binomial <- fillers_5upkids %>%
  group_by(participant) %>%
  summarise(
    successes = sum(response.corr),
    trials = length(response.corr)
  ) %>%
  rowwise() %>%
  mutate(
    binom_test = list(binom.test(successes, trials, p=0.5, alternative="greater")),
    p_value = binom_test$p_value
  ) %>%
  ungroup()
print(fillers_5upkids_binomial$binom_test)
 

############################
#practice scores
############################
## do same analyses as for fillers
#get practice scores by participant
practicescore_5up <- ddply(practice_5up, .(participant), summarise, practice_score=mean(response.corr))

#add this calculation to the original aduchi dataframe
aduchi_5up <- merge(aduchi_5up, practicescore_5up, by="participant")

#average practice scores by age group
practicescorebyage_5up <- ddply(practice_5up, .(participant, age_y), summarise, practice_score=mean(response.corr))
practicescorebyage_5up$age_grp <- revalue(practicescorebyage_5up$age_y, c("6"="5"))
meanpracticescore_5up <- ddply(practicescorebyage_5up, .(age_grp), summarise, practice_score=mean(practice_score))

#are practice scores above chance in all age groups?
#5-year-olds: yes
t.test(practicescorebyage_5up$practice_score[practicescorebyage_5up$age_grp==5], mu=.5)
#adults: yes
t.test(practicescorebyage_5up$practice_score[practicescorebyage_5up$age_grp=="adu"], mu=.5)

## Try a binomial test for the practice items 
#   (basically ignores individual participant contribution to total)
# Need to group the practice_5up data by participant and perform a binomial test
# adults
practice_5upadults = subset(practice_5up, age_y == "adu")
practice_5upkids = subset(practice_5up, age_y != "adu")

practice_5upadults_binomial <- practice_5upadults %>%
  group_by(participant) %>%
  summarise(
    successes = sum(response.corr),
    trials = length(response.corr)
  ) %>%
  rowwise() %>%
  mutate(
    binom_test = list(binom.test(successes, trials, p=0.5, alternative="greater")),
    p_value = binom_test$p_value
  ) %>%
  ungroup()
print(practice_5upadults_binomial$binom_test)

practice_5upkids_binomial <- practice_5upkids %>%
  group_by(participant) %>%
  summarise(
    successes = sum(response.corr),
    trials = length(response.corr)
  ) %>%
  rowwise() %>%
  mutate(
    binom_test = list(binom.test(successes, trials, p=0.5, alternative="greater")),
    p_value = binom_test$p_value
  ) %>%
  ungroup()
print(practice_5upkids_binomial$binom_test)

#########################
#Proportion subject responses
#########################  
  #########
  #data prep for graphs
  #########
  # add new column for morphology favors subject to aduchi_5up dataframe
  # morph_favors = subject if number == subject_num, else object
  aduchi_5up$morph_favors <- ifelse(aduchi_5up$number == aduchi_5up$subject_num, "sub", "obj")

  #get individual participants' proportion subject responses 
  # in each combination of connective, form and target antecedent, across pl and sg
  subresponse_5up <- ddply(aduchi_5up, 
                           .(participant, item, age_y, age_m, type, congruence, 
                             connective, nullovert, number, subject_num, 
                             morph_favors,
                             baseline_bias), summarise, response.subj=mean(response.subj))
  subresponse_5up$age_grp <- revalue(subresponse_5up$age_y, c("6"="5"))  

  # Get condition means and 95\% confidence intervals instead of SE
  # setting random seed for bootstrapping
  set.seed(613)
  # mean function for use in calculating bootstrapped CIs
  mean_function <- function(data, indices) {
    return(mean(data[indices], na.rm = TRUE))
  }
  # Function to compute CIs 
  bootstrap_ci <- function(data, n_boot = 10000) { #10,000 iterations
    boot_result <- boot(data = data, 
                        statistic = mean_function, 
                        R = n_boot)
    ci <- boot.ci(boot_result, type = "perc")
    
    return(list(
      mean = mean(data),
      # 95% range
      lower = ci$percent[4],
      upper = ci$percent[5]
    ))
  }
  
  
  propsubj_5up <- ddply(subresponse_5up, 
                        .(age_grp, 
                          morph_favors,
                          connective, nullovert), 
                        summarise, 
                        Subject_responses=mean(response.subj), 
                        # CIs
                        CI_lower = bootstrap_ci(response.subj)$lower,
                        CI_upper = bootstrap_ci(response.subj)$upper
                        )
  

  #get individual participants' proportion subject responses in each combination of connective, form and target antecedent, across pl and sg
  #Get condition means and CIs
 
  propsubjnum_5up <- ddply(subresponse_5up, 
                           .(age_grp, 
                             morph_favors,
                             connective, nullovert, number), 
                           summarise, 
                           Subject_responses=mean(response.subj), 
                           # CIs
                           CI_lower = bootstrap_ci(response.subj)$lower,
                           CI_upper = bootstrap_ci(response.subj)$upper
                           )
  
  #########
  #Figures: Graph proportion subjects in each condition and age group + split out by number
  #########
  
  ####format morphology variable 
  propsubj_5up$morph_favors <- as.factor(propsubj_5up$morph_favors)
  #order subjects before objects
  propsubj_5up$morph_favors <- relevel(propsubj_5up$morph_favors, "sub")
  #replace 'sub' and 'obj' with fully spelled out names
  propsubj_5up$morph_favors <- revalue(propsubj_5up$morph_favors, c("sub"="subject", "obj"="non-subject"))
  
  ####format morphology variable
  propsubjnum_5up$morph_favors <- as.factor(propsubjnum_5up$morph_favors)
  #order subjects before objects
  propsubjnum_5up$morph_favors <- relevel(propsubjnum_5up$morph_favors, "sub")
  #replace 'sub' and 'obj' with fully spelled out names
  propsubjnum_5up$morph_favors <- revalue(propsubjnum_5up$morph_favors, c("sub"="subject", "obj"="non-subject"))
  
  # convert number to unordered factor
  propsubjnum_5up$number <- as.factor(propsubjnum_5up$number)
  #order singular before plural
  propsubjnum_5up$number <- relevel(propsubjnum_5up$number, "sg")
  
  
  ####format age group variable
  #get participant age ranges and Ns
  age5NandR_5up <- paste0("≥5\n(", partNandR_5up$Min_ym[partNandR_5up$age_y == 5],"-",partNandR_5up$Max_ym[partNandR_5up$age_y == 5], ", N=", partNandR_5up$N[partNandR_5up$age_y == 5], ")")
  aduNandR_5up <- paste0("adults\n(N=", partNandR_5up$N[partNandR_5up$age_y == 'adu'], ")")
  #create a named character vector for use in age group facet labels
  age.labs <- c(age5NandR_5up, aduNandR_5up)
  names(age.labs) <- c("5", "adu")
  #optional: create a single sub-title with all this information together
  partchar_5up = paste0("Age groups: 5 and up (", partNandR_5up$Min_ym[partNandR_5up$age_y == 5],"-",partNandR_5up$Max_ym[partNandR_5up$age_y == 5], ", N=", partNandR_5up$N[partNandR_5up$age_y == 5], "), adults (N=", partNandR_5up$N[partNandR_5up$age_y == 'adu'], ")")
  
  ####format nullovert variable
  #create a named character vector for use in pronominal form facet labels
  nullovert.labs <- c("null (favors subject)", "overt (favors non-subject)")
  names(nullovert.labs) <- c("null", "overt")
  

  #plot proportion subject responses, by condition, for each age group
  ggplot(propsubj_5up, 
         aes(x=morph_favors,
         y=Subject_responses, 
         fill=connective)) +
    facet_grid(nullovert~age_grp, labeller = labeller(age_grp = age.labs, nullovert = nullovert.labs)) +
    geom_bar(stat="identity", position="dodge", color="#339999") +
          geom_errorbar(aes(ymin = CI_lower, ymax = CI_upper),
                      width=0.25, position=position_dodge(.9)) +
    geom_abline(slope = 0, intercept = .5, linetype="dashed") +
    scale_fill_manual(name="Connective", values=c("#339999", "#AAC0C0"),
                      breaks=c("despues", "porque"), 
                      labels=c("después (favors subject)", "porque (favors non-subject)")) +
    labs(title = "Subject antecedent responses", 
         subtitle = "Error bars represent 95% CIs",
         x="Antecedent favored by morphology", 
         y="Proportion subject antecedent responses") +
    theme_bw() +
    theme(legend.position="right")

  #save hi-res picture of this graph for paper (re)submission
  ggsave("adu-5up-subresp-12hx16w-bg.png", dpi=800, dev='png', height=12, width=16, units="cm") 
  
  
  ### plot separated out by morphology type (sg vs. pl)
  #########################
  #Plot subject responses BY NUMBER
  #########################  
  # adults vs. 5+ kids in two separate plots
  
  #adults
  adults <- propsubjnum_5up %>% filter(age_grp == 'adu')
  ggplot(adults, 
         aes(x=morph_favors,
             y=Subject_responses, fill=connective)) +
    facet_grid(nullovert~number, labeller = labeller(nullovert = nullovert.labs)) +
    geom_bar(stat="identity", position="dodge", color="dark blue") +
    geom_errorbar(aes(ymin = CI_lower, ymax = CI_upper),
                  width=0.25, 
                  position=position_dodge(.9)) +
    geom_abline(slope = 0, intercept = .5, linetype="dashed") +
    scale_fill_manual(name="Connective", values=c("dark blue", "dark grey"),
                      breaks=c("despues", "porque"), 
                      labels=c("después (favors subject)", "porque (favors non-subject)")) +
    labs(title = "Subject antecedent responses by adults, sg vs. pl", 
         subtitle = "Error bars represent 95% CIs",
         x="Antecedent favored by morphology", 
         y="Proportion subject antecedent responses") +
    theme_bw() +
    theme(legend.position="right")
  
  #save hi-res picture of this graph for inclusion in appendix
  ggsave("adu-subrespnum-12hx16w-bg.png", dpi=800, dev='png', height=10, width=16, units="cm")
  
  #5yo
  fiveup <- propsubjnum_5up %>% filter(age_grp == '5')
  ggplot(fiveup, 
         aes(x=morph_favors,
             y=Subject_responses, fill=connective)) +
    facet_grid(nullovert~number, labeller = labeller(nullovert = nullovert.labs)) +
    geom_bar(stat="identity", position="dodge", color="dark green") +
    geom_errorbar(aes(ymin = CI_lower, ymax = CI_upper),
                  width=0.25, position=position_dodge(.9)) +
    geom_abline(slope = 0, intercept = .5, linetype="dashed") +
    scale_fill_manual(name="Connective", values=c("dark green", "dark grey"),
                      breaks=c("despues", "porque"), 
                      labels=c("después (favors subject)", "porque (favors non-subject)")) +
    labs(title = "Subject antecedent responses by 5-year-olds, sg vs. pl", 
         subtitle = "Error bars represent 95% CIs",
         x="Antecedent favored by morphology", 
         y="Proportion subject antecedent responses") +
    theme_bw() +
    theme(legend.position="right")
  
  #save hi-res picture of this graph for inclusion in appendix
  ggsave("5up-subjrespnum--12hx16w-bg.png", dpi=800, dev='png', height=10, width=16, units="cm")
  

### Now try to do it split out by sg, adult vs 5up and pl, adult vs. 5up  
# sg   
propsubjnumsg_5up <- propsubjnum_5up %>% filter(number == 'sg')
  ggplot(propsubjnumsg_5up, 
         aes(x=morph_favors,      
             y=Subject_responses, fill=connective)) +
    facet_grid(nullovert~age_grp, labeller = labeller(age_grp = age.labs, nullovert = nullovert.labs)) +
    geom_bar(stat="identity", position="dodge", color="violet") +
    geom_errorbar(aes(ymin = CI_lower, ymax = CI_upper),
                  width=0.25, position=position_dodge(.9)) +
    geom_abline(slope = 0, intercept = .5, linetype="dashed") +
    scale_fill_manual(name="Connective", values=c("violet", "grey"),
                      breaks=c("despues", "porque"), 
                      labels=c("después (favors subject)", "porque (favors subject)")) +
    labs(title = "Subject antecedent responses, sg morphology", 
         subtitle = "Error bars represent 95% CIs",
         x="Antecedent favored by morphology", 
         y="Proportion subject antecedent responses") +
    theme_bw() +
    theme(legend.position="right")

  #save hi-res picture of this graph for inclusion in appendix
  ggsave("adu-5up-sg-subjresp--12hx16w-bg.png", dpi=800, dev='png', height=10, width=16, units="cm")
  
    
# pl
  propsubjnumpl_5up <- propsubjnum_5up %>% filter(number == 'pl')
  ggplot(propsubjnumpl_5up, 
         aes(x=morph_favors, 
             y=Subject_responses, fill=connective)) +
    facet_grid(nullovert~age_grp, labeller = labeller(age_grp = age.labs, nullovert = nullovert.labs)) +
    geom_bar(stat="identity", position="dodge", color="purple") +
    geom_errorbar(aes(ymin = CI_lower, ymax = CI_upper),
                  width=0.25, position=position_dodge(.9)) +
    geom_abline(slope = 0, intercept = .5, linetype="dashed") +
    scale_fill_manual(name="Connective", values=c("purple", "grey"),
                      breaks=c("despues", "porque"), 
                      labels=c("después (favors subject)", "porque (favors non-subject)")) +
    labs(title = "Subject antecedent responses, pl morphology", 
         subtitle = "Error bars represent 95% CIs",
         x="Antecedent favored by morphology", 
         y="Proportion subject antecedent responses") +
    theme_bw() +
    theme(legend.position="right")
  
  #save hi-res picture of this graph for inclusion in appendix
  ggsave("adu-5up-pl-subjresp--12hx16w-bg.png", dpi=800, dev='png', height=10, width=16, units="cm")

  #########
  #data prep for models
  #########
  #refactor the conditions as numeric, with subject-favoring conditions = 1 and object-favoring conditions = 0: 
  # + (morphology) number with 'sg' = 1 and 'pl' = 0
    # morph_favors refers to whether morphology on the verb favors the subject or the object of the preceding sentence
  aduchi_5up$morph_favors_numeric <- as.numeric(paste(revalue(aduchi_5up$morph_favors, c("sub"=1, "obj"=0))))
    # connective despues favors the preceding subject while porque favors the preceding object
  aduchi_5up$connective_numeric <- as.numeric(paste(revalue(aduchi_5up$connective, c("despues"=1, "porque"=0))))
    # null subject pronoun favors the preceding subject while overt does not
  aduchi_5up$nullovert_numeric <- as.numeric(paste(revalue(aduchi_5up$nullovert, c("null"=1, "overt"=0))))
   # number has 'sg' = 1, `pl` = 0
  aduchi_5up$number_numeric <- as.numeric(paste(revalue(aduchi_5up$number, c("sg"=1, "pl"=0))))  
  
  # separate adults from children as a categorical factor
  aduchi_5up$age_grp <- as.numeric(paste(revalue(aduchi_5up$age_y, c("5"=0, "6"=0, "adu"=1))))

  # recommended: contrast coding of all variables of interest. 
  # From Sonderegger 2023: Helmert coding because centered & orthogonal
  aduchi_5up$morph_favors_factor <- as.factor(aduchi_5up$morph_favors_numeric)
  aduchi_5up$number_factor <- as.factor(aduchi_5up$number_numeric) # pl = 1, sg = 2
  aduchi_5up$connective_factor <- as.factor(aduchi_5up$connective_numeric) 
  aduchi_5up$nullovert_factor <- as.factor(aduchi_5up$nullovert_numeric) 
  aduchi_5up$age_grp_factor <- as.factor(aduchi_5up$age_grp)   
  #aduchi_5up_morph_favors_contrast <- contrasts(aduchi_5up$morph_favors_factor)
  contrasts(aduchi_5up$morph_favors_factor) <- contr.helmert(2)
  contrasts(aduchi_5up$number_factor) <- contr.helmert(2)
  contrasts(aduchi_5up$connective_factor) <- contr.helmert(2)
  contrasts(aduchi_5up$nullovert_factor) <- contr.helmert(2)
  contrasts(aduchi_5up$age_grp_factor) <- contr.helmert(2)
    
  #########
  #Modeling Q1: Which main effects are significant at each age (5, adult)?
  ######### 
  # Big model with age_grp as fixed factor (so one big model, 
  #        with relevant interactions included)
  #        fixed effects: form (nullovert), connective, morphology (morph_favors), age (age_grp)
  #        potential interactors from visual inspection: morphology number (number)
  #        random effects: item, participant
  
  aduchi_5up.bigmodelbasicglmer <- glmer(response.subj ~ 
                                           nullovert_factor # fixed effect form
                                         + connective_factor # fixed effect connective
                                         + morph_favors_factor # fixed effect morphology
                                         + age_grp # fixed effect age
                                         + (1|item)  # by-item random intercept
                                         + (1|participant),  # by-participant random intercept 
                                         data = aduchi_5up,
                                       family = binomial(link="logit"),
                                       glmerControl(optimizer = "bobyqa"))
  summary(aduchi_5up.bigmodelbasicglmer) 
  
  ## add in fixed effect of morphology number and compare
  aduchi_5up.bigmodelbasicwithnumglmer <- update(aduchi_5up.bigmodelbasicglmer,
                                            . ~ . + number_factor)
  summary(aduchi_5up.bigmodelbasicwithnumglmer)
  anova(aduchi_5up.bigmodelbasicglmer,aduchi_5up.bigmodelbasicwithnumglmer, refit = FALSE)
  # didn't help on its own (p=.7707)
  
  # add in interaction of age with other fixed effects
  aduchi_5up.bigmodelbasicageinterglmer <- update(aduchi_5up.bigmodelbasicglmer,
                                                  . ~ . 
                                                  + nullovert_factor:age_grp
                                                  + connective_factor:age_grp
                                                  + morph_favors_factor:age_grp
                                                  )
  summary(aduchi_5up.bigmodelbasicageinterglmer)
  anova(aduchi_5up.bigmodelbasicglmer,aduchi_5up.bigmodelbasicageinterglmer, refit = FALSE)
  # improved from basic: p = 1.64e-14
  
  # add in interaction with morphology number with other fixed effects
  aduchi_5up.bigmodelbasicnuminterglmer <- update(aduchi_5up.bigmodelbasicglmer,
                                                  . ~ . 
                                                  + nullovert_factor:number_factor
                                                  + connective_factor:number_factor
                                                  + morph_favors_factor:number_factor
                                                  + age_grp:number_factor
  )
  summary(aduchi_5up.bigmodelbasicnuminterglmer)
  anova(aduchi_5up.bigmodelbasicglmer,aduchi_5up.bigmodelbasicnuminterglmer, refit = FALSE)
  # improved from basic: p =.000671
  
  # compare age_grp and num interaction additions
  anova(aduchi_5up.bigmodelbasicageinterglmer,aduchi_5up.bigmodelbasicnuminterglmer, refit = FALSE)
  # age better than number interaction in terms of AIC, BIC, and logL, but ChiSq weird (=0) 
  
  # add in interactions of number on top of interactions age
  aduchi_5up.bigmodelbasicagenuminterglmer <- update(aduchi_5up.bigmodelbasicageinterglmer,
                                                  . ~ . 
                                                  + nullovert_factor:age_grp:number_factor
                                                  + connective_factor:age_grp:number_factor
                                                  + morph_favors_factor:age_grp:number_factor
  )
  summary(aduchi_5up.bigmodelbasicagenuminterglmer)
  # compare age interaction alone to age+number interactions
  anova(aduchi_5up.bigmodelbasicageinterglmer,aduchi_5up.bigmodelbasicagenuminterglmer, refit = FALSE)
  # didn't help enough: p =.5196 of being better
  
  # compare age+number interactions against basic
  anova(aduchi_5up.bigmodelbasicglmer,aduchi_5up.bigmodelbasicagenuminterglmer, refit = FALSE)
  # improved over basic: p = 1.169e-12
  
  # FIXED EFFECTS takeaway: Just interaction by age (like Morgan originally suggested)
  # aduchi_5up.bigmodelbasicageinterglmer
  
  # Explore random effects structure
  # fixed effects to investigate: nullovert, connective, morph_favors, age_grp, number
  # variance across participants (specific to each participant, but doesn't vary within participant): 
  #            (information use) nullovert, connective, morph_favors, number
  # variance across (linguistic) items (specific to each item, but doesn't vary within item): 
  #           none of these except morphology number? 
  #           nullovert, connective, morph_favors are already included in the fixed effects
  
  # basic fixed effects with age, only random intercepts for participant and item
  aduchi_5up.basicageinterglmer <- aduchi_5up.bigmodelbasicageinterglmer
  # let's check item first (number)
  aduchi_5up.ageinteritemnumglmer <- update(aduchi_5up.basicageinterglmer,
                                            . ~ .
                                            - (1|item) # remove this one
                                            + (1 + number_factor|item) # add in that number affects item especially
                                            )
  summary(aduchi_5up.ageinteritemnumglmer)
  # compare item random with morphology number vs without
  anova(aduchi_5up.basicageinterglmer, aduchi_5up.ageinteritemnumglmer)
  # didn't help enough: p = .803 of not better
  
  # what if we squish all the variance by item into the number factor interaction?
  aduchi_5up.ageinteritemnumonlyglmer <- update(aduchi_5up.basicageinterglmer,
                                            . ~ .
                                            - (1|item) # remove this one
                                            + (0 + number_factor|item) # add in that number affects item especially
  )
  summary(aduchi_5up.ageinteritemnumonlyglmer)
  # compare item random with morphology number vs without
  anova(aduchi_5up.basicageinterglmer, aduchi_5up.ageinteritemnumonlyglmer)
  # didn't help enough: p = .803 of not better
  
  # Conclusion: just go with random intercept by item (1 | item)
  
  # now let's check participant-level
  # fixed effects alone, then interactions
  # nullovert
  aduchi_5up.ageinterpartformglmer <- update(aduchi_5up.basicageinterglmer,
                                             . ~ .
                                             - (1|participant) # remove this one
                                             + (1 + nullovert_factor|participant) # add in that form affects participant especially
  )
  summary(aduchi_5up.ageinterpartformglmer)
  anova(aduchi_5up.basicageinterglmer, aduchi_5up.ageinterpartformglmer)
  # helped over basic: p = .013
  
  # connective
  aduchi_5up.ageinterpartconnglmer <- update(aduchi_5up.basicageinterglmer,
                                             . ~ .
                                             - (1|participant) # remove this one
                                             + (1 + connective_factor|participant) # add in that connective affects participant especially
  )
  summary(aduchi_5up.ageinterpartconnglmer) # isSingular warning
  # check out the random effects variance-covariance matrix
  # It often indicates a convergence issue or an overparameterized model (likely this one).
  print(VarCorr(aduchi_5up.ageinterpartconnglmer), comp=c("Variance", "Std.Dev."))
  anova(aduchi_5up.basicageinterglmer, aduchi_5up.ageinterpartconnglmer)  
  # helped over basic, but may be unrealible: p = 2.177e-05
  anova(aduchi_5up.ageinterpartformglmer, aduchi_5up.ageinterpartconnglmer)    
  # helped over form, but may be unreliable: p = 0
  ### takeaway: don't include connective_factor | participant as comparison
  
  # morph_favors
  aduchi_5up.ageinterpartmorphglmer <- update(aduchi_5up.basicageinterglmer,
                                             . ~ .
                                             - (1|participant) # remove this one
                                             + (1 + morph_favors_factor|participant) # add in that morph affects participant especially
  )
  summary(aduchi_5up.ageinterpartmorphglmer)
  anova(aduchi_5up.basicageinterglmer, aduchi_5up.ageinterpartmorphglmer)
  # helped over basic: p = 6.518e-10
  # compare to form and (potentially unreliable) connective
  anova(aduchi_5up.ageinterpartformglmer, aduchi_5up.ageinterpartmorphglmer)  
  # morph has lower AIC, BIC, and logL, but ChiSq gives back a 0 so a little unclear
  #anova(aduchi_5up.ageinterpartconnglmer, aduchi_5up.ageinterpartmorphglmer)    
  # morph helped more
  
  # number
  aduchi_5up.ageinterpartnumglmer <- update(aduchi_5up.basicageinterglmer,
                                              . ~ .
                                              - (1|participant) # remove this one
                                              + (1 + number_factor|participant) # add in that number affects participant especially
  )
  
  # isSingular warning, so unreliable (may be overparameterized)
  summary(aduchi_5up.ageinterpartnumglmer)
  anova(aduchi_5up.basicageinterglmer, aduchi_5up.ageinterpartnumglmer)
  # didn't help over basic: p = .1037
  # compare to form, (potentially unreliable) connective, and morph
  anova(aduchi_5up.ageinterpartformglmer, aduchi_5up.ageinterpartnumglmer)  
  # didn't help more than form
  anova(aduchi_5up.ageinterpartconnglmer, aduchi_5up.ageinterpartnumglmer)    
  # didn't help more than conn
  anova(aduchi_5up.ageinterpartmorphglmer, aduchi_5up.ageinterpartnumglmer) 
  # didn't help more than morph
  ## takeaway: don't compare
    
  # nullovert*connective (unreliable) *morph_favors* num (unreliable)
  # so just check nullovert and morph_favors
  aduchi_5up.ageinterpartformmorphglmer <- update(aduchi_5up.basicageinterglmer,
                                            . ~ .
                                            - (1|participant) # remove this one
                                            + (1 + nullovert_factor:morph_favors_factor|participant) # add in that form and morph affect participant especially
  ) 
  # isSingular warning (may be overparameterized)
  summary(aduchi_5up.ageinterpartformmorphglmer)
  # this actually makes parameterization worse, but I wanted to check it
  aduchi_5up.ageinterpartformmorphglmer2 <- update(aduchi_5up.basicageinterglmer,
                                                  . ~ .
                                                  - (1|participant) # remove this one
                                                  + (1 + nullovert_factor*morph_favors_factor|participant) # add in that form and morph affect participant especially
  ) 
  # isSingular warning (may be overparameterized)
  summary(aduchi_5up.ageinterpartformmorphglmer2)
  
  # compare to basic, form alone, and morph alone
  anova(aduchi_5up.basicageinterglmer, aduchi_5up.ageinterpartformmorphglmer)
  # helped over basic, p=2.19e-05
  anova(aduchi_5up.ageinterpartformglmer, aduchi_5up.ageinterpartformmorphglmer)  
  # helped over form alone, p=.000155
  anova(aduchi_5up.ageinterpartmorphglmer, aduchi_5up.ageinterpartformmorphglmer)  
  # didn't help over form alone, p=.9762
  
  # nullovert:connective
  aduchi_5up.ageinterpartformconnglmer <- update(aduchi_5up.basicageinterglmer,
                                                 . ~ .
                                                 - (1|participant) # remove this one
                                                 + (1 + nullovert_factor:connective_factor|participant) # add in that morph & number affect participant especially
  ) 
  # isSingular warning
  summary(aduchi_5up.ageinterpartformconnglmer)
  # nullovert:num
  aduchi_5up.ageinterpartformnumglmer <- update(aduchi_5up.basicageinterglmer,
                                                 . ~ .
                                                 - (1|participant) # remove this one
                                                 + (1 + nullovert_factor:number_factor|participant) # add in that morph & number affect participant especially
  ) 
  # failed to converge warning
  summary(aduchi_5up.ageinterpartformnumglmer)  
  
  # connective:morph
  aduchi_5up.ageinterpartconnmorphglmer <- update(aduchi_5up.basicageinterglmer,
                                                . ~ .
                                                - (1|participant) # remove this one
                                                + (1 + connective_factor:morph_favors_factor|participant) # add in that morph & number affect participant especially
  ) 
  # isSingular warning
  summary(aduchi_5up.ageinterpartconnmorphglmer)  
    
  # connective:num
  aduchi_5up.ageinterpartconnnumglmer <- update(aduchi_5up.basicageinterglmer,
                                                  . ~ .
                                                  - (1|participant) # remove this one
                                                  + (1 + connective_factor:number_factor|participant) # add in that morph & number affect participant especially
  ) 
  # isSingular warning
  summary(aduchi_5up.ageinterpartconnnumglmer)  
    
  # morph:num
  aduchi_5up.ageinterpartmorphnumglmer <- update(aduchi_5up.basicageinterglmer,
                                                  . ~ .
                                                  - (1|participant) # remove this one
                                                  + (1 + morph_favors_factor:number_factor|participant) # add in that morph & number affect participant especially
  ) 
  # maximum number of function evaluations exceeded -- try with bigger limit
  summary(aduchi_5up.ageinterpartmorphnumglmer)
  aduchi_5up.ageinterpartmorphnumglmerbiggerlim <- update(aduchi_5up.ageinterpartmorphnumglmer,
                                                          . ~ .,
                                                  (control = 
                                                     glmerControl(optimizer = "bobyqa",
                                                       optCtrl = list(maxfun = 100000)))
                                                   )  
  # max number of function evaluations still exceeded
  summary(aduchi_5up.ageinterpartmorphnumglmerbiggerlim)
  ## Takeaway: random effects structure too complicated
  
  
    # Takeaway: Just add random effect for participant by morph_favors
  
  # final big model = basic+age_grp interactions with (1|item) and (1+morph_favors|participant)
  summary(aduchi_5up.ageinterpartmorphglmer)  
  
  #####
  ##########
  # separate out by age, because age interacts significantly with form and morph_favors
  # 5+ kids subgroup
  # start with estimating fixed effects
  # basic = form, connective, morphology, random slopes for item and participant
  chi5up.aduchi_5up.basicglmer <- glmer(response.subj ~ 
                                           nullovert_factor # fixed effect form
                                         + connective_factor # fixed effect connective
                                         + morph_favors_factor # fixed effect morphology
                                         + (1|item)  # by-item random intercept
                                         + (1|participant),  # by-participant random intercept 
                                         data = subset(aduchi_5up, age_grp == 0),
                                         family = binomial(link="logit"),
                                         glmerControl(optimizer = "bobyqa"))
  summary(chi5up.aduchi_5up.basicglmer) 
  
  ### morph number checks, because by visual inspection, that seemed to matter
  ## add in fixed effect of morphology number and compare
  chi5up.aduchi_5up.basicwithnumglmer <- update(chi5up.aduchi_5up.basicglmer,
                                                 . ~ . + number_factor)
  summary(chi5up.aduchi_5up.basicwithnumglmer)
  anova(chi5up.aduchi_5up.basicglmer,chi5up.aduchi_5up.basicwithnumglmer, refit = FALSE)
  # didn't help over basic: p = .72
  
  # add in interaction with morphology number with other fixed effects
  chi5up.aduchi_5up.basicnuminterglmer <- update(chi5up.aduchi_5up.basicglmer,
                                                  . ~ . 
                                                  + number_factor
                                                  + nullovert_factor:number_factor
                                                  + connective_factor:number_factor
                                                  + morph_favors_factor:number_factor
  )
  summary(chi5up.aduchi_5up.basicnuminterglmer)
  anova(chi5up.aduchi_5up.basicglmer,chi5up.aduchi_5up.basicnuminterglmer, refit = FALSE)
  # improved from basic: p = 3.087e-6
  
  # what if just the significant interaction of morph:number
  chi5up.aduchi_5up.basicmorphnuminterglmer <- update(chi5up.aduchi_5up.basicglmer,
                                                 . ~ . 
                                                 + number_factor
                                                 + morph_favors_factor:number_factor
  )
  summary(chi5up.aduchi_5up.basicmorphnuminterglmer)
  anova(chi5up.aduchi_5up.basicglmer,chi5up.aduchi_5up.basicmorphnuminterglmer, refit = FALSE)
  # improved over basic: p = 6.348e-7
  anova(chi5up.aduchi_5up.basicnuminterglmer,chi5up.aduchi_5up.basicmorphnuminterglmer, refit = FALSE)  
  # not better than including all interactions with number (p=.2949)
  
  # FIXED EFFECTS takeaway: basic+number interactions 
  # (but could just do basic+morph:num if convergence issues)
  ###  to minimize parameters, doing basic+morph:num
  # chi5up.aduchi_5up.basicmorphnuminterglmer
  
  # now check random effects
  # Explore random effects structure
  # fixed effects to investigate: nullovert, connective, morph_favors, number
  # variance across participants (specific to each participant, but doesn't vary within participant): 
  #            (information use) nullovert, connective, morph_favors, number
  # variance across (linguistic) items (specific to each item, but doesn't vary within item): 
  #           none of these?
  #           nullovert, connective, morph_favors, number interaction are already included in the fixed effects
  # so should just be (1|item)
  # check participants
  # nullovert
  chi5up.aduchi_5up.morphnuminterpartformglmer <- update(chi5up.aduchi_5up.basicmorphnuminterglmer,
                                             . ~ .
                                             - (1|participant) # remove this one
                                             + (1 + nullovert_factor|participant) # add in that form affects participant especially
  )
  # singular warning (unreliable)
  summary(chi5up.aduchi_5up.morphnuminterpartformglmer)
  #anova(chi5up.aduchi_5up.basicnuminterglmer, chi5up.aduchi_5up.numinterpartformglmer)
  
  # connective
  chi5up.aduchi_5up.morphnuminterpartconnglmer <- update(chi5up.aduchi_5up.basicmorphnuminterglmer,
                                                    . ~ .
                                                    - (1|participant) # remove this one
                                                    + (1 + connective_factor|participant) # add in that connective affects participant especially
  )  
  summary(chi5up.aduchi_5up.morphnuminterpartconnglmer)
  anova(chi5up.aduchi_5up.basicmorphnuminterglmer, chi5up.aduchi_5up.morphnuminterpartconnglmer)
  # helps over basic: p = .02699 
    
  # morph_favors
  chi5up.aduchi_5up.morphnuminterpartmorphglmer <- update(chi5up.aduchi_5up.basicmorphnuminterglmer,
                                                    . ~ .
                                                    - (1|participant) # remove this one
                                                    + (1 + morph_favors_factor|participant) # add in that morph_favors affects participant especially
  )  
  summary(chi5up.aduchi_5up.morphnuminterpartmorphglmer)
  anova(chi5up.aduchi_5up.basicmorphnuminterglmer, chi5up.aduchi_5up.morphnuminterpartmorphglmer)
  # helps over basic: p = .002029 
  anova(chi5up.aduchi_5up.morphnuminterpartconnglmer, chi5up.aduchi_5up.morphnuminterpartmorphglmer)
  # may help over connective: AIC, BIC, log likelihood lower, but ChiSq returns empty value      
  
  # number
  chi5up.aduchi_5up.morphnuminterpartnumglmer <- update(chi5up.aduchi_5up.basicmorphnuminterglmer,
                                                          . ~ .
                                                          - (1|participant) # remove this one
                                                          + (1 + number_factor|participant) # add in that number affects participant especially
  )  
  # isSingular warning (unreliable)
  summary(chi5up.aduchi_5up.morphnuminterpartnumglmer)
  
  # individually, both conn and morph help over basic, morph moreso based on non-ChiSq metrics
  
  # check interactions
  # form:conn
  chi5up.aduchi_5up.morphnuminterpartformconnglmer <- update(chi5up.aduchi_5up.basicmorphnuminterglmer,
                                                        . ~ .
                                                        - (1|participant) # remove this one
                                                        + (1 + nullovert_factor:connective_factor|participant) # add in that form & connective affects participant especially
  )  
  # didn't converge
  summary(chi5up.aduchi_5up.morphnuminterpartformconglmer)
  
  # form:morph
  chi5up.aduchi_5up.morphnuminterpartformmorphglmer <- update(chi5up.aduchi_5up.basicmorphnuminterglmer,
                                                             . ~ .
                                                             - (1|participant) # remove this one
                                                             + (1 + nullovert_factor:morph_favors_factor|participant) # add in that form & morph affects participant especially
  )  
  # isSingular warning
  summary(chi5up.aduchi_5up.morphnuminterpartformmorphglmer)  
  
  # form:num
  chi5up.aduchi_5up.morphnuminterpartformnumglmer <- update(chi5up.aduchi_5up.basicmorphnuminterglmer,
                                                              . ~ .
                                                              - (1|participant) # remove this one
                                                              + (1 + nullovert_factor:number_factor|participant) # add in that form & num affects participant especially
  )  
  # isSingular warning
  summary(chi5up.aduchi_5up.morphnuminterpartformnumglmer) 
  
  # conn:morph
  chi5up.aduchi_5up.morphnuminterpartconnmorphglmer <- update(chi5up.aduchi_5up.basicmorphnuminterglmer,
                                                            . ~ .
                                                            - (1|participant) # remove this one
                                                            + (1 + connective_factor:morph_favors_factor|participant) # add in that connective & morph affects participant especially
  )  
  # isSingular warning
  summary(chi5up.aduchi_5up.morphnuminterpartconnmorphglmer) 
  
  # conn:num
  chi5up.aduchi_5up.morphnuminterpartconnumglmer <- update(chi5up.aduchi_5up.basicmorphnuminterglmer,
                                                              . ~ .
                                                              - (1|participant) # remove this one
                                                              + (1 + connective_factor:number_factor|participant) # add in that connective & morph num affects participant especially
  )  
  # isSingular warning
  summary(chi5up.aduchi_5up.morphnuminterpartconnumglmer)   
  
  # morph:num
  chi5up.aduchi_5up.morphnuminterpartmorphnumglmer <- update(chi5up.aduchi_5up.basicmorphnuminterglmer,
                                                           . ~ .
                                                           - (1|participant) # remove this one
                                                           + (1 + morph_favors_factor:number_factor|participant) # add in that morph & morph num affects participant especially
  )  
  # isSingular warning
  summary(chi5up.aduchi_5up.morphnuminterpartmorphnumglmer)     
  
  ## Takeaway: interactions involve overparameterization
  ## Bigger takeaway: morph alone, so (1 + morph_favors_factor|participant)
  ##   chi5up.aduchi_5up.morphnuminterpartmorphglmer
  
  summary(chi5up.aduchi_5up.morphnuminterpartmorphglmer)

  
### now for adults
  
  #########
  # adult subgroup
  # start with estimating fixed effects
  # basic = form, connective, morphology, random slopes for item and participant
  adu.aduchi_5up.basicglmer <- glmer(response.subj ~ 
                                          nullovert_factor # fixed effect form
                                        + connective_factor # fixed effect connective
                                        + morph_favors_factor # fixed effect morphology
                                        + (1|item)  # by-item random intercept
                                        + (1|participant),  # by-participant random intercept 
                                        data = subset(aduchi_5up, age_grp == 1),
                                        family = binomial(link="logit"),
                                        glmerControl(optimizer = "bobyqa"))
  summary(adu.aduchi_5up.basicglmer) 
  # isSingular warning....
  
  ## add in fixed effect of morphology number and compare (since we did this with 5+ kids)
  adu.aduchi_5up.basicwithnumglmer <- update(adu.aduchi_5up.basicglmer,
                                                . ~ . + number_factor)
  summary(adu.aduchi_5up.basicwithnumglmer)
  # isSingular warning....
  anova(adu.aduchi_5up.basicglmer,adu.aduchi_5up.basicwithnumglmer, refit = FALSE)
  # didn't help over basic: p = .57
  
  # add in interaction with morphology number with other fixed effects
  adu.aduchi_5up.basicnuminterglmer <- update(adu.aduchi_5up.basicglmer,
                                                 . ~ . 
                                                 + number_factor
                                                 + nullovert_factor:number_factor
                                                 + connective_factor:number_factor
                                                 + morph_favors_factor:number_factor
  )
  summary(adu.aduchi_5up.basicnuminterglmer)
  anova(adu.aduchi_5up.basicglmer,adu.aduchi_5up.basicnuminterglmer, refit = FALSE)
  # didn't help: 0.517
  
  # FIXED EFFECTS takeaway: basic interactions 
  # adu.aduchi_5up.basicglmer
  
  # now check random effects
  # Explore random effects structure
  # fixed effects to investigate: nullovert, connective, morph_favors, number
  # variance across participants (specific to each participant, but doesn't vary within participant): 
  #            (information use) nullovert, connective, morph_favors, number
  # variance across (linguistic) items (specific to each item, but doesn't vary within item): 
  #           none of these?
  #           nullovert, connective, morph_favors, number interaction are already included in the fixed effects
  # so should just be (1|item)
  # check participants
  # nullovert
  adu.aduchi_5up.basicpartformglmer <- update(adu.aduchi_5up.basicglmer,
                                                         . ~ .
                                                         - (1|participant) # remove this one
                                                         + (1 + nullovert_factor|participant) # add in that form affects participant especially
  )
  # singular warning (unreliable)
  summary(adu.aduchi_5up.basicpartformglmer)
  # failed to converge
  #anova(chi5up.aduchi_5up.basicnuminterglmer, chi5up.aduchi_5up.numinterpartformglmer)
  
  # connective
  adu.aduchi_5up.basicpartconnglmer <- update(adu.aduchi_5up.basicglmer,
                                                         . ~ .
                                                         - (1|participant) # remove this one
                                                         + (1 + connective_factor|participant) # add in that connective affects participant especially
  )  
  summary(adu.aduchi_5up.basicpartconnglmer)
  # isSingular
 
  
  # morph_favors
  adu.aduchi_5up.basicpartmorphglmer <- update(adu.aduchi_5up.basicglmer,
                                                          . ~ .
                                                          - (1|participant) # remove this one
                                                          + (1 + morph_favors_factor|participant) # add in that morph_favors affects participant especially
  )  
  summary(adu.aduchi_5up.basicpartmorphglmer)
  anova(adu.aduchi_5up.basicglmer, adu.aduchi_5up.basicpartmorphglmer)
  # helps over basic: p = 6.462e-7 

  # number
  adu.aduchi_5up.basicpartnumglmer <- update(adu.aduchi_5up.basicglmer,
                                                        . ~ .
                                                        - (1|participant) # remove this one
                                                        + (1 + number_factor|participant) # add in that number affects participant especially
  )  
  # isSingular warning (unreliable)
  summary(adu.aduchi_5up.basicpartnumglmer)
  
  # individually, morph helps over basic
  # adu.aduchi_5up.basicpartmorphglmer
  
  # check interactions
  # form:conn
  adu.aduchi_5up.basicpartformconnglmer <- update(adu.aduchi_5up.basicglmer,
                                                             . ~ .
                                                             - (1|participant) # remove this one
                                                             + (1 + nullovert_factor:connective_factor|participant) # add in that form & connective affects participant especially
  )  
  # didn't converge
  summary(adu.aduchi_5up.basicpartformconglmer)
  
  # form:morph
  adu.aduchi_5up.basicpartformmorphglmer <- update(adu.aduchi_5up.basicglmer,
                                                              . ~ .
                                                              - (1|participant) # remove this one
                                                              + (1 + nullovert_factor:morph_favors_factor|participant) # add in that form & morph affects participant especially
  )  
  # isSingular warning
  summary(adu.aduchi_5up.basicpartformmorphglmer)  
  
  # form:num
  adu.aduchi_5up.mbasicpartformnumglmer <- update(adu.aduchi_5up.basicglmer,
                                                            . ~ .
                                                            - (1|participant) # remove this one
                                                            + (1 + nullovert_factor:number_factor|participant) # add in that form & num affects participant especially
  )  
  # isSingular warning
  summary(adu.aduchi_5up.mbasicpartformnumglmer) 
  
  # conn:morph
  adu.aduchi_5up.basicpartconnmorphglmer <- update(adu.aduchi_5up.basicglmer,
                                                              . ~ .
                                                              - (1|participant) # remove this one
                                                              + (1 + connective_factor:morph_favors_factor|participant) # add in that connective & morph affects participant especially
  )  
  # isSingular warning
  summary(adu.aduchi_5up.basicpartconnmorphglmer ) 
  
  # conn:num
  adu.aduchi_5up.basicpartconnumglmer <- update(adu.aduchi_5up.basicglmer,
                                                           . ~ .
                                                           - (1|participant) # remove this one
                                                           + (1 + connective_factor:number_factor|participant) # add in that connective & morph num affects participant especially
  )  
  # isSingular warning
  summary(adu.aduchi_5up.basicpartconnumglmer)   
  
  # morph:num
  adu.aduchi_5up.basicpartmorphnumglmer <- update(adu.aduchi_5up.basicglmer,
                                                             . ~ .
                                                             - (1|participant) # remove this one
                                                             + (1 + morph_favors_factor:number_factor|participant) # add in that morph & morph num affects participant especially
  )  
  # isSingular warning
  summary(adu.aduchi_5up.basicpartmorphnumglmer)     
  
  ## Takeaway: interactions involve overparameterization
  ## Bigger takeaway: morph alone, so (1 + morph_favors_factor|participant)
  ##   adu.aduchi_5up.basicpartmorphglmer
  
  summary(adu.aduchi_5up.basicpartmorphglmer)
  

  ## Do emmeans for estimating beta value difference between 5+ kids and adults

# combined model includes age interaction:aduchi_5up.ageinterpartmorphglmer

  combined_model <- aduchi_5up.ageinterpartmorphglmer
  # from ZotGPT
# form (null/overt) differed by age so 
emm_null <- emmeans(combined_model, ~ nullovert_factor | age_grp)
contrast(emm_null, interaction = "pairwise", by = NULL)
# significant: adults sensitive 1.11 log-odds times as much as kids
# 1. Get simple effects (null/overt effect within each age group)
#emm_forminage <- emmeans(combined_model, ~ nullovert_factor | age_grp)
pairs(emm_null)
#pairs(emm_forminage)
# adults 2.5 times higher odds of positive response for overt forms

# morphology differed by age, so 
emm_morph <- emmeans(combined_model, ~ morph_favors_factor | age_grp)
contrast(emm_morph, interaction = "pairwise", by = NULL)
# significant: both groups sensitive but adults moreso than kids
#emm_morphinage <- emmeans(combined_model, ~ morph_favors_factor | age_grp)
pairs(emm_morph)
#pairs(emm_morphinage)
# adults massively more sensitive than kids

# what about age in general for adults vs children?
emm_age <- emmeans(combined_model, ~ age_grp)
contrast(emm_age, interaction = "pairwise", by = NULL)
# p=.07...not clearly significant

