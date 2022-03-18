# SIMULATED DATA

library(NetworkRiskMeasures)
library(readxl)
library(dplyr)

# Setting Seed
set.seed(4400)

# GET BANK DATA

# read data from excel
df <- read_excel('./data/data.xlsx', sheet="filtered_2017") %>%
  select('Net Loans to Banks', 'Total Deposits from Banks', 'Total Capital') %>%
  mutate_if(is.character,as.numeric) %>%
  na.omit %>%
  filter(.['Net Loans to Banks'] > 50 & .['Total Deposits from Banks'] > 50 & .['Total Capital'] > 0)

# get interbank assets
assets <- df[['Net Loans to Banks']]

# get interbank liabilitites
liabilities <- df[['Total Deposits from Banks']]

# Making sure assets = liabilities
assets <- sum(liabilities) * (assets/sum(assets))
# liabilities <- sum(assets) * (liabilities/sum(liabilities))

# get capital buffer
buffer <- df[['Total Capital']]
# buffer <- pmax(0.01, runif(length(liabilities), min = 0.01, max = 0.05) * (assets + liabilities))

# Weights as a function of assets and liabilities
weights <- assets + liabilities

# creating data.frame
sim_data <- data.frame(bank  = paste0("b", 1:length(liabilities)),
                       assets = assets,
                       liabilities = liabilities,
                       buffer = buffer,
                       weights = weights,
                       buffer_p = (buffer / liabilities))


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
