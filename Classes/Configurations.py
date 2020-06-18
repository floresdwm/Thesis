# Global script configurations

# Statistical confidence level 95% = 1.98; 99% = 3;
confidence_pca = 3
outlier_confidence_level_x = 3
outlier_confidence_level_y = 3

# Sigma outliers
sigma_detection = False
sigma_percentage = False
sigma_confidence = 8

# Polarization test, not working yet
polarization_test = False
polarization_n_groups = 2

# Train-test data split percentage  0.7 = 70% to train
train_split_percentage = 0.7

# MISC Inform pre-processing transformation method
transformation = 'SNV'  # modelos> "SNV", "Primeira derivada (5pts)", "SNV + Primeira derivada (5pts)", "SNV + Primeira derivada (7pts)"
Xinit = 1
Xend = 117
scans = 500