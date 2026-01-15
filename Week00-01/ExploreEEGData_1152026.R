library(readr)
library(dplyr)

s_of_t_subset <- read_csv("s_of_t_subset.csv",
     col_names = c("s"), # Changed from FALSE to specify column name
     col_types = cols(S = col_double())
) # Updated to match new column name


# Rename s to value
# s_of_t_subset <- s_of_t_subset %>% rename(value = s)

# Sampling frequency
fs <- 500

# Create a time column
s_of_t_subset <- s_of_t_subset %>% mutate(time = (row_number() - 1) / fs)
# s_of_t_subset = s_of_t_subset[,c(2,1)]
View(s_of_t_subset)

# Plot the data
plot(s_of_t_subset$time, s_of_t_subset$s,
     main = "EEG Signal Depicting Seizure",
     xlab = "Time (seconds)", ylab = "Signal (s)", type = "l"
)

# write the data to a csv file
write.csv(s_of_t_subset, "s_of_t_subset_withTimeAxis.csv", row.names = FALSE)

## Lets add a column with Normalized values of the signal, for each time-point
s_of_t_subset$normalizedValue <- (s_of_t_subset$s - mean(s_of_t_subset$s)) / sd(s_of_t_subset$s)
# Plot the data
plot(s_of_t_subset$time, s_of_t_subset$normalizedValue,
     main = "Normalized EEG Signal Depicting Seizure",
     xlab = "Time (seconds)", ylab = "Signal (s)", type = "l"
)

plot(density(s_of_t_subset$s), main = "Kernel Probability Density Function (unnormalized)")
lines(density(s_of_t_subset$normalizedValue), col = "red")

plot(density(s_of_t_subset$normalizedValue), col = "red", main = "Kernel Probability Density Function (normalized)")

s_of_t_subset$GT <- ifelse(s_of_t_subset$time > 12, "Seizure", "Normal") # NOTE: 12s and above was seizure.
s_of_t_subset$GT <- as.factor(s_of_t_subset$GT)

names(s_of_t_subset)[1] <- "value"
write.csv(s_of_t_subset, file = "s_of_t_subset_CLEAN.csv", row.names = F)
