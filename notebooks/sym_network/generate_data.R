# SIMULATED DATA

library(NetworkRiskMeasures)

# GENERATE BANK DATA

# number of banks
n <- 1500

# Setting Seed
set.seed(4400)

# Heavy tailed assets
assets <- rlnorm(n, 0, 2)
assets[assets < 4] <- runif(length(assets[assets < 4]))

# Heavy tailed liabilities
liabilities <- rlnorm(n, 0, 2) 
liabilities[liabilities < 4] <- runif(length(liabilities[liabilities < 4]))

# Making sure assets = liabilities
assets <- sum(liabilities) * (assets/sum(assets))

# Buffer as a function of liabilities
buffer <- pmax(0.01, runif(length(liabilities))*liabilities + abs(rnorm(n, 4, 2.6)))

# Weights as a function of assets, buffer and liabilities
weights <- (assets + liabilities + buffer + 1) + rlnorm(n, 0, 1)

# creating data.frame
sim_data <- data.frame(bank  = paste0("b", 1:n),
                       assets = assets,
                       liabilities = liabilities,
                       buffer = buffer,
                       weights = weights)


# GENERATE NETWORK

# maximum entropy
#md_mat <- matrix_estimation(sim_data$assets, sim_data$liabilities, method = "me", verbose = F)

# minimum density estimation
# seed - min. dens. estimation is stochastic
md_mat <- matrix_estimation(sim_data$assets, sim_data$liabilities, method = "md", verbose = F)

# rownames and colnames for the matrix
rownames(md_mat) <- colnames(md_mat) <- sim_data$bank


#PLOT

library(ggplot2)
library(ggnetwork)
library(igraph)

# converting our network to an igraph object
gmd <- graph_from_adjacency_matrix(md_mat, weighted = T)

# adding other node attributes to the network
V(gmd)$buffer <- sim_data$buffer
V(gmd)$weights <- sim_data$weights/sum(sim_data$weights)
V(gmd)$assets  <- sim_data$assets
V(gmd)$liabilities <- sim_data$liabilities

# ploting with ggplot and ggnetwork
netdf <- ggnetwork(gmd)

ggplot(netdf, aes(x = x, y = y, xend = xend, yend = yend)) +
  geom_edges(arrow = arrow(length = unit(6, "pt"), type = "closed"),
             color = "grey50", curvature = 0.1, alpha = 0.5) +
  geom_nodes(aes(size = weights)) +
  ggtitle("Estimated interbank network") +
  theme_blank()


# CONTAGION

# The DebtRank methodology proposed by Bardoscia et al (2015) considers a linear shock propagation
contdr <- contagion(exposures = md_mat, buffer = sim_data$buffer, weights = sim_data$weights, shock = "all", method = "debtrank", verbose = F)
summary(contdr)
#plot(contdr)


# a bank may not transmit contagion unless it defaults -> contagion method to threshold.
# Traditional default cascades simulation
# contthr <-  contagion(exposures = md_mat, buffer = sim_data$buffer, weights = sim_data$weights, 
#                       shock = "all", method = "threshold", verbose = F)
# summary(contthr)


# The contagion() function is flexible and you can simulate arbitrary scenarios with it. For example, 
# how would simultaneous stress shocks of 1% up to 25% in all banks affect the system? To do that, 
# just create a list with the shock vectors and pass it to contagion().
# s <- seq(0.01, 0.25, by = 0.01)
# shocks <- lapply(s, function(x) rep(x, nrow(md_mat)))
# names(shocks) <- paste(s*100, "pct shock")
# cont <- contagion(exposures = gmd, buffer = sim_data$buffer, shock = shocks, weights = sim_data$weights, method = "debtrank", verbose = F)
# summary(cont)
# plot(cont, size = 2.2)


# SAVE DATA

contdr_table <- summary(contdr)$summary_table
colnames(contdr_table)[1] <- colnames(sim_data)[1]

# combine all data
merged = merge(sim_data,contdr_table, by = "bank")

# write csv
write.csv(md_mat, "network.csv", row.names = TRUE)
write.csv(merged, "nodes.csv", row.names = TRUE)




# plot summary
# smtable = contdr_table
smtable = contdr_table[contdr_table['additional_stress'] < 0.02,
                       1:ncol(contdr_table)]
colnames(smtable)[1] = 'scenario'
q = quantile(merged$additional_stress)

labels = FALSE
nudge_y = 0.01
check_overlap = TRUE
size = 3 
color = "black"

leg_0 = paste("75% (0): ",round(q[[4]],5))
leg_1 = paste("50% (1): ",round(q[[3]],5))
leg_2 = paste("25% (2): ",round(q[[2]],5))
leg_3 = paste("0% (3): ",round(q[[1]],5))

p <- ggplot(smtable, aes_string("original_stress", "additional_stress", label = "scenario")) + 
  geom_point(shape = 21) +
  geom_hline(aes(yintercept=q[[1]], linetype = leg_3), color='purple') +
  geom_hline(aes(yintercept=q[[2]], linetype = leg_2), color='orange') +
  geom_hline(aes(yintercept=q[[3]], linetype = leg_1), color='red') +
  geom_hline(aes(yintercept=q[[4]], linetype = leg_0), color='blue') +
  guides(linetype=guide_legend(title="Percentile (label): value"))

if (labels) {
  p <- p + geom_text(nudge_y = nudge_y,
                     check_overlap = check_overlap,
                     size = size,
                     color = color)
}

p <- p + 
  xlab("\nOriginal Stress") +
  ylab("Additional Stress\n") +
  ggtitle("Original Stress vs Additional Stress\n") +
  aes_string(fill = "additional_defaults") +
  scale_fill_gradient(name = "Additional\nDefaults", 
                      low = "green",
                      high = "red") +
  theme_minimal() 
p
