# Make a model that predicts the normalized value of a signal using the un-normalized value of the signal
# normalized signal = signal - mean(signal) / sd(signal)
# normalized signal = f(signal) ... this is the regression model
# Lets assume the regrsesion model f(signal) = beta_1*signal + beta_0
# beta_1 and beta_0 are the parameters of the model that we need to estimate
# We can estimate these parameters using the least squares method
# write the regression fit in R

library(readr)
s_of_t_subset_CLEAN <- read_csv("s_of_t_subset_CLEAN_with2SDrule.csv",
    col_types = cols(GT = col_factor(levels = c(
        "Normal",
        "Seizure"
    )), TwoSDRule = col_factor(levels = c(
        "InControl",
        "OutOfControl"
    )))
)
View(s_of_t_subset_CLEAN)

fit <- lm(normalizedValue ~ value, data = s_of_t_subset_CLEAN)

print(summary(fit))


# Now lets try and estimate the response of GT using normalizedValue and original value, with logistic regression
# Convert normalizedValue to binary (0/1)
s_of_t_subset_CLEAN$GT_binary <- ifelse(s_of_t_subset_CLEAN$GT == "Seizure", 1, 0)

# Fit logistic regression model
model <- glm(GT_binary ~ normalizedValue, # + value,
    data = s_of_t_subset_CLEAN,
    family = binomial(link = "logit")
)

# View summary of the model
summary(model)

# Optional: Calculate odds ratios
odds_ratios <- exp(coef(model))
print("Odds Ratios:")
print(odds_ratios)
