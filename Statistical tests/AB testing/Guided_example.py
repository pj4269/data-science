# The experiment:  BlueBird company has been using the same advertising email for years, and some at the company are starting to feel that it's getting a little stale. Ever the data driven individual, you propose an experiment. The marketing department draws up a new version of the email, and you'll conduct an A/B test comparing the two emails.

# To do list before the experiment: 
# 1. Have a copy of Both Emails. This is useful so that anyone looking into the experiment can use the research proposal to actually see what happens.
# 2. Roll out plan: how long will the experiment last? how many people will be treated?
# Often times, though 50:50 rollouts are the fastest, a gradual rollout will be used to ensure that there are no negative consequences of the experiment or if things are broken or severely degraded it can be rolled back quickly. In this case we'll do two weeks at 10% and then two weeks at 50%. That will lead to about 30,000+ in treatment, which is a nice sized sample

#3. Success Metric: Signups (also called conversions), and use t-test with a p-value of .05

#4. Secondary metric: The time from email to signup.

#5. Test the infrastructure before rolling out. In this case that means making sure the email is set up properly to be sent out when the experiment turns on. If it's manual, that's pretty easy (make sure it's formatted properly and the email is generally deliverable). If you use a service, sending some test emails to test accounts is probably a good idea. This can be either an engineering or a data science task, depending on complexity.

#6. Method for randomly sampling subjects, and know if people stay in test or control permanently(meaning if they will be tested once). This can matter for something like emails if people receive multiple messages. Should they only receive the new style of email or should each email be random? For simplicity, we'll assume each individual is only emailed once.

#7. Make sure your experiment doesn't collide with other events or experiments. If you were testing marketing strategies for football betting behavior right before the Super Bowl, for example, you might see some weird behaviors that would make your findings not broadly applicable. You also don't want to test two things that are related at the same time.

#8. Another consideration is segmentation of your sample. Sometimes you don't want to test on the entire population. Maybe you'll pick specific cities (in which case it may be appropriate to sample those in a representative fashion) or test a specific age group or tenure of user. Make these decisions early as they will help inform later analysis and possibly how the experiment is functionally set up. For here we'll just assume we're small enough to test on the entire population.

#9. Finally, it is good practice to give everyone a chance to review the research proposal and provide feedback

# Let's say we flipped the switch and turned the experiment on at 10% on April 1, 2016.







