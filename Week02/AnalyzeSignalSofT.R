library(readr)
library(dplyr)

# Read the CSV file
library(readr)
s_of_t_subset_CLEAN <- read_csv("s_of_t_subset_CLEAN.csv",
        col_types = cols(GT = col_factor(levels = c(
                "Normal",
                "Seizure"
        )))
)
View(s_of_t_subset_CLEAN)

## Create a box plot of the signal for eachh cclass of the GT column using base R
boxplot(s_of_t_subset_CLEAN$normalizedValue ~ s_of_t_subset_CLEAN$GT,
        xlab = "Class", ylab = "Signal",
        main = "Boxplot of Signal by Class"
)


# Lets compare the means of the two classes using t-test
t.test(s_of_t_subset_CLEAN$normalizedValue ~ s_of_t_subset_CLEAN$GT)

# lets try a non-parametric test like Wilcoxon rank sum test
wilcox.test(s_of_t_subset_CLEAN$normalizedValue ~ s_of_t_subset_CLEAN$GT)
plot(density(s_of_t_subset_CLEAN$normalizedValue[s_of_t_subset_CLEAN$GT == "Normal"]), col = "blue", main = "EEG Signal Depicting Seizure", xlab = "Normalized Signal", ylab = "Density")
lines(density(s_of_t_subset_CLEAN$normalizedValue[s_of_t_subset_CLEAN$GT == "Seizure"]), col = "red", main = "EEG Signal Depicting Seizure", xlab = "Normalized Signal", ylab = "Density")


## In-control v/s out of control visualization

# Plot the normalized signal data
plot(s_of_t_subset_CLEAN$time, s_of_t_subset_CLEAN$normalizedValue,
        main = "Mean-SD Normalized EEG Signal Depicting Seizure",
        xlab = "Time (seconds)", ylab = "Signal (s)", type = "l"
)

# out of control signal
s_of_t_subset_CLEAN_out_of_control <- s_of_t_subset_CLEAN

# if s_of_t_subset_CLEAN_out_of_control$normalizedValue < 2 or >-2 then it is in control.  Drop these rows
s_of_t_subset_CLEAN_out_of_control <- s_of_t_subset_CLEAN_out_of_control[s_of_t_subset_CLEAN_out_of_control$normalizedValue > 2 | s_of_t_subset_CLEAN_out_of_control$normalizedValue < -2, ]

# superimpse the out of control signal as red stars on the plot
points(s_of_t_subset_CLEAN_out_of_control$time, s_of_t_subset_CLEAN_out_of_control$normalizedValue, col = "red", pch = 8)

# Lets put the 2 SD rule signal into the main data frame
s_of_t_subset_CLEAN$TwoSDRule <- as.factor(ifelse(s_of_t_subset_CLEAN$normalizedValue > 2 |
        s_of_t_subset_CLEAN$normalizedValue < -2, "OutOfControl", "InControl"))

table(s_of_t_subset_CLEAN$GT, s_of_t_subset_CLEAN$TwoSDRule)

write.csv(s_of_t_subset_CLEAN, "s_of_t_subset_CLEAN_with2SDrule.csv", row.names = F)


# Lets change the 2 SD rule to a 1 SD rule

# out of control signal
s_of_t_subset_CLEAN_out_of_control <- s_of_t_subset_CLEAN

# if s_of_t_subset_CLEAN_out_of_control$normalizedValue < 2 or >-2 then it is in control.  Drop these rows
s_of_t_subset_CLEAN_out_of_control <- s_of_t_subset_CLEAN_out_of_control[s_of_t_subset_CLEAN_out_of_control$normalizedValue > 1 |
        s_of_t_subset_CLEAN_out_of_control$normalizedValue < -1, ]

# superimpse the out of control signal as red stars on the plot
points(s_of_t_subset_CLEAN_out_of_control$time, s_of_t_subset_CLEAN_out_of_control$normalizedValue, col = "red", pch = 8)

# Lets put the 2 SD rule signal into the main data frame
s_of_t_subset_CLEAN$OneSDRule <- as.factor(ifelse(s_of_t_subset_CLEAN$normalizedValue > 1 |
        s_of_t_subset_CLEAN$normalizedValue < -1, "OutOfControl", "InControl"))

table(s_of_t_subset_CLEAN$GT, s_of_t_subset_CLEAN$OneSDRule)
