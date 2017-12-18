library(tensorflow)
library(tidyverse)
library(reshape2)

set.seed(1)

### PART 1: Data input and preprocessing
creditcard <- read_csv(file = "./data/creditcard.csv.zip", 
                       col_names = TRUE,
                       col_types =  cols(.default = "n"))
creditcard <- creditcard %>%  mutate(Class=factor(Class, labels=c("valid","fraud")))


# Transform Amount
#     Amount: log10 of Amount + 0.01 (minimum unit of currency)
creditcard <- creditcard %>% mutate(Amount = log10(Amount+0.01))
# Add 24-hour Time cycle variable and rescale it
creditcard <- creditcard %>% mutate(TimeCycle = Time - 86400*(Time %/% 86400))

# Scale non-PCA variables

a <- min(select(creditcard,V1:V28))
b <- max(select(creditcard,V1:V28))

creditcard2 <- creditcard %>% mutate_at(vars(Time, TimeCycle, Amount),
                                       function(x){
                                           a+(b-a)*(x - min(x))/(max(x)- min(x))
                                        })





### PART 2: Assign data to train/dev/test groups

# Count exaMples in each class
m_fraud <- sum(creditcard$Class == "fraud")
m_valid <- sum(creditcard$Class == "valid")

# Determine numbers of valid transactions to split 
# across train/dev/test sets 33%/33%/33%
m_valid_33pc <- m_legal %/% 3
m_valid_rem <- m_legal %% 3

m_valid_train <- m_valid_33pc + m_valid_rem
m_valid_dev <-  m_valid_33pc
m_valid_test <- m_valid_33pc

# Determine numbers of fraudulent transactions to 
# split across dev/test sets 50%/50%
m_fraud_50pc <- round(m_fraud *0.5)

m_fraud_dev <- m_fraud_50pc
m_fraud_test <- m_fraud - m_fraud_50pc

# Arrange data by Class 0 -> 1 (legal -> fraud)
creditcard <- arrange(creditcard, Class)

# Create randomised set vectors for legal and fraudulent transactions
set_fraud <- sample(c( rep("dev", m_fraud_dev),
                       rep("test" , m_fraud_test)))
set_legal <- sample(c( rep("train", m_valid_train),
                       rep("dev" , m_valid_dev),
                       rep("test" , m_valid_test)))
# Concatenate assignment vectors colinearly to arranged data (fraud -> legal )
# and add as new column to the data
set_vec <- c(set_legal, set_fraud)
creditcard <- mutate(creditcard, Set=set_vec)

### PART 3: Do sanity check and print train/dev/test split
# Check there are no fraudulent transactions assigned to the train set
fraud_in_train_set <- creditcard$Set[creditcard$Class=="fraud"] == "train"

if(any(fraud_in_train_set)){ 
    stop("Found fraudulent transactions in train set")
}else{
    rm(fraud_in_train_set)
}

# Print train/dev/test split
train_dev_test <- creditcard %>% 
    mutate(Class=if_else(Class==1, "fraud", "legal")) %>%
    group_by(Set, Class) %>%
    count 
print(train_dev_test)
creditcard <- ungroup(creditcard)

### PART 4: Split sets
# Create and shuffle training set
train_set <- creditcard %>% filter(Set=="train") %>% sample_frac

# Create dev and test sets (no need to shuffle)
dev_set <- creditcard %>% filter(Set=="dev")
test_set <- creditcard %>% filter(Set=="test")

# Separate X and Y
train <- list( X = tibble(), Y = tibble() )
dev <- list( X = tibble(), Y = tibble() )
test <- list( X = tibble(), Y = tibble() )

train$X <- train_set %>% select(-Class, -Set)
train$Y <- train_set %>% select(Class, -Set)
dev$X <- dev_set %>% select(-Class, -Set)
dev$Y <- dev_set %>% select(Class, -Set)
test$X <- test_set %>% select(-Class, -Set)
test$Y <- test_set %>% select(Class, -Set)

# Cleanup unnecessary large objects
rm(set_vec, set_legal, set_fraud, creditcard, train_set, dev_set, test_set)

# Put test set aside during training
save(test, file="test.Rdata")
save(train, file="train.Rdata")
save(dev, file="dev.Rdata")
rm(test)

# Dimensions X: m * n_x
# Dimensions Y: m * n_y

## DEFINE AND INITIALISE PARAMETER TENSORS W, b
# Arg: 
#     layer_sizes: a vector of layer sizes including input and output layers
# Ret: 
#    A nested list of weight (W) and bias (b) tensors indexed by layer number
initialise_parameters <- function(layer_sizes){
    W <- list()
    b <- list()
    
    # Define tensors for parameters of L layers
    # L = number of **hidden** layers (i.e. not including input)
    L <- length(layer_sizes)-1
    for (i in 1:L){
        
        dim_0 <- layer_sizes[i]
        dim_1 <- layer_sizes[i+1]
        
        Wi_name <- paste0("W", i)
        bi_name <- paste0("b", i)
        
        # Initialise weights with Xavier initialisation
        W[[i]] <- tf$get_variable(
            name = Wi_name, 
            shape = c(dim_0, dim_1), 
            initializer = tf$contrib$layers$xavier_initializer())
        # Initialise bias with zeros
        b[[i]] <-  tf$get_variable(
            name = bi_name, 
            shape = c(1, dim_1), 
            initializer = tf$zeros_initializer())
    }
    return(list( W=W, b=b))
}
restore_parameters <- function(pars_file){
    load(pars_file)
    W <- list()
    b <- list()
    
    for (i in seq_along(pars$W)){
        tW <- tf$get_variable(name = paste0("W",i),
                              shape = dim(pars$W[[i]]),
                              initializer = tf$constant_initializer(
                                  value=pars$W[[i]],
                                  dtype = tf$float32))
        W <- c(W,tW)
    }
    
    for (i in seq_along(pars$b)){
        tb <- tf$get_variable(name = paste0("b",i),
                              shape = dim(pars$b[[i]]),
                              initializer = tf$constant_initializer(
                                  value=pars$b[[i]],
                                  dtype = tf$float32))
        b <- c(b,tb)
    }
    return(list( W=W, b=b))
}
## DEFINE FORWARD PROPAGATION USING TANH LAYERS
# Args:
#     X: tensor of predictor variables, with dimensionss [m, n_x]
#     parameters: a nested list of parameter tensors indexed by layer number
#     keep_prob: a scalar tensor for 1-dropout probability
# Ret:
#     The linear output Z of the final layer

forward_prop <- function(X, parameters,keep_prob){
    # Unpack parameters
    W <- parameters$W
    b <- parameters$b
    
    
    L <- length(W) # Number of hidden layers
    
    Z <- list()  # Linear result
    a <- list()  # Activations before dropout
    A <- list()  # activations after dropout
    
    # Run first layer with X as input 
    Z[[1]] <- tf$matmul(X, W[[1]]) + b[[1]]
    a[[1]] <- tf$nn$relu(Z[[1]])
    A[[1]] <- tf$nn$dropout(a[[1]], keep_prob)
    
    # If we have more than 1 hidden layer...
    if(L>2){
        # Loop through them using activations A of previous layers
        for(i in 2:(L-1)){
            Z[[i]] <- tf$matmul(A[[i-1]],W[[i]]) + b[[i]]
            a[[i]] <- tf$nn$relu(Z[[i]])
            A[[i]] <- tf$nn$dropout(a[[i]], keep_prob)
        }
    }
    # Calculate linear output for final layer
    Z[[L]] <- tf$matmul(A[[L-1]],W[[L]]) + b[[L]]
    
    return(Z[[L]])
}

## DEFINE MODEL TRAINING FUNCTION
# Args:
#     train, dev: training and dev sets, as lists of X and Y tibbles
#     w: overrepresentation factor of whitenoise vs actual examples
# Ret:
#     parameters as a list of base R matrices (not tensors)
train_model <- function(train, dev,
                        learning_rate = 0.0001, 
                        num_epochs = 100, 
                        examples_per_minibatch = 100,
                        w=1,
                        seed=1,
                        keep=1,
                        pars_file=NULL,
                        ...){
    args <- paste(match.call(), collapse=" ")
 
    tf$reset_default_graph()  # Reset tensorflow graph
    tf$set_random_seed(seed)  # Set seed for tensorflow
    set.seed(seed)            # Set seed for base R
    
    # Create log for current run
    tstamp <- as.integer(Sys.time())
    
    argsfile <- paste0("args_",tstamp)
    logdir <- file.path( getwd(), paste0("run_",tstamp) )
    dir.create(logdir)
    
    write_file(x=args, path = file.path(logdir,argsfile))
    
    
    # Number of examples
    m <- nrow(train$X)
    # Number of input variables
    n_x <- ncol(train$X)
    # Number of response variables
    n_y <- ncol(train$Y)
    # Layer sizes (note: including input AND output layers)
    layer_sizes = c(n_x, n_x, n_x, n_x, n_y)
    
    ###############################
    ### Tensor graph definition ###
    ###############################
    # Input, output and dropout  placeholders
    X <- tf$placeholder(dtype = tf$float32, shape = shape(NULL,n_x))
    Y <- tf$placeholder(dtype = tf$float32, shape = shape(NULL,n_y))
    keep_prob = tf$placeholder(dtype=tf$float32)
    
    # Whitenoise generator - makes network sceptical
    # "w" argument is excess of whitenoise over signal 
    min <- min(train$X)
    max <- max(train$X)
    m_w <- w*examples_per_minibatch
    whitenoise <- tf$random_uniform(shape=shape(m_w,n_x),
                                    minval = min,
                                    maxval = max,
                                    dtype = tf$float32)
    # Initialise parameters.
    parameters <- list()
    if(is.null(pars_file)){
        parameters <- initialise_parameters(layer_sizes)
    }else{
        parameters <- restore_parameters(pars_file)
    }
    
    # Define forward propagation
    Y_hat <- forward_prop(X, parameters,keep_prob)
    
    # Define predicion (sigmoid), loss and cost
    pred <- tf$nn$sigmoid(Y_hat)
    loss <- tf$losses$log_loss(Y,pred)
    cost <- tf$reduce_mean(loss)
    
    # Define optimiser
    optimiser <-  tf$train$AdamOptimizer(learning_rate = learning_rate)$minimize(cost)
    
    # Define initialiser
    init <- tf$global_variables_initializer()
    ############################
    ### End graph definition ###
    ############################
    
    cost_log <- tibble(epoch=numeric(),cost=numeric())
    stats_log <-tibble(epoch=numeric(),
                       AUC=numeric(),
                       TPR_eq_PPV=numeric(),
                       F1=numeric())
    
    
    
    # Start tensorflow session
    sess <- tf$Session()
    sess$run(fetches=init)
    secs_last_epoch <- 0
    best_AUC <- 0
    best_F1 <- 0
    for(epoch in 1:num_epochs){
        
        cat("\nEpoch ",epoch,"/",num_epochs,"\n")
        epoch_cost <- 0
        bat_i <- 1
        
        # Generate a shuffle vector of size m for the train set
        shuffle_train <- sample(1:m,size = m, replace = FALSE)
        
        # Number of batches
        n_batches <- ceiling( m/examples_per_minibatch )
        time <- Sys.time()
        batch_digits <- ceiling(log10(n_batches))
        
        for(i in seq(from=1, to=m, by=examples_per_minibatch )){
            
            # First member of minibatch
            a <- i
            # Last member of minibatch [ min() accounts for final batch ]
            z <- min(i+examples_per_minibatch-1, m)
            
            # Shuffle train X and Y by shuffle vector
            # "Signal" contains only transactions labelled as valid (Class 0)
            signal_X <- train$X[shuffle_train[a:z],,drop=FALSE]
            signal_Y <- train$Y[shuffle_train[a:z],,drop=FALSE]
           
            # Generate "Noise" set. Label noise as fraudulent (Class 1).
            FRAUD <- 1
            noise_X <- as.tibble(sess$run(whitenoise))[1:(w*nrow(signal_X)),]
            noise_Y <- as.tibble(matrix(FRAUD,nrow=w*nrow(signal_Y) ,ncol=n_y ))
            
            # Noise sets need the same names as signal sets to be 
            # efficiently joined using dplyr::bind_rows()
            names(noise_Y) <- names(signal_Y)
            names(noise_X) <- names(signal_X)
            
            # Actual minibatch size, including noise set.
            mb_size <- nrow(signal_X)+nrow(noise_X)
            
            # Generate a shuffle vector of mb_size
            shuffle_mb <- sample(1:mb_size,size=mb_size,replace = FALSE)
            
            # Bind signal and noise, sort by shuffle_mb, cast into matrix.
            batch_X <- bind_rows(signal_X, noise_X)[shuffle_mb,] %>% as.matrix()
            batch_Y <- bind_rows(signal_Y,noise_Y)[shuffle_mb,] %>% as.matrix()
            
            # Run optimiser and cost on batch_X/batch_Y with dropout
            
            sess_data <- sess$run(
                fetches = list(optimiser,cost),
                feed_dict = dict(X = batch_X, Y=batch_Y,keep_prob=keep))
            batch_cost <- sess_data[[2]]
            epoch_cost <- epoch_cost + batch_cost/n_batches
            
            form_str <- paste0("%",batch_digits,".f")
            del_str <- paste0(rep("\r",1+2*batch_digits))
            
            cat("\tProcessing batch ", sprintf(form_str,bat_i))
            cat("/")
            cat(sprintf(form_str,n_batches))
            cat(del_str)
            bat_i <- bat_i+1
        }
        cost_log <- cost_log %>% add_row(epoch=epoch, cost=epoch_cost)
        cat(paste0("\nCost = ", epoch_cost,"\t\n\n"))
        
       
        dev_pred <- sess$run(pred,feed_dict=dict(X=as.matrix(dev$X),keep_prob=1))
        
        # Prepare roc thresholds
        roc <- tibble(threshold = seq(0, 1, 0.005))
        
        positives <- sapply(dev_pred,function(y){y > roc$threshold})
        negatives <- !positives
        
        fraud <- matrix(dev$Y == 1, length(roc$threshold),length(dev_pred),byrow = TRUE)
        legal <- !fraud
        
        True_Pos <- rowSums(positives & fraud)
        True_Neg <- rowSums(negatives & legal)
        False_Pos <- rowSums(positives & legal)
        False_Neg <- rowSums(negatives & fraud)
        
        # TPR: true positive rate, recall, sensitivity
        # FPR: false positive rate
        # PPV: positive predictive value, precision
        roc <- roc %>% mutate(TPR = True_Pos/(True_Pos+False_Neg),
                              FPR = False_Pos/(False_Pos+True_Neg),
                              PPV = True_Pos/(True_Pos+False_Pos),
                              F1  = 2*(PPV*TPR)/(PPV+TPR))
        
        # Calculate AUC for current ROC curve
        height <- 0.5*(roc$TPR[-1] + roc$TPR[-length(roc$TPR)])
        width <- -diff(roc$FPR)
        AUC <- sum(height*width)
        # Determine index of maximum F1 in current ROC curve
        i_maxF1 <- which.max(roc$F1)

        roc_title <- paste0("Epoch ",epoch,": AUC = ", sprintf("%.3f",AUC),".\t",
                            "Best dev threshold ", roc$threshold[[i_maxF1]],":")
        
        roc_subtitle <- paste0("F1 = ", sprintf("%.3f",roc$F1[[i_maxF1]]),"; ",
                               "TPR = ",sprintf("%3.2f",100*roc$TPR[[i_maxF1]]),"%; ",
                               "FPR = ",sprintf("%3.2f",100*roc$FPR[[i_maxF1]]),"%; ",
                               "PPV = ",sprintf("%3.2f",100*roc$FPR[[i_maxF1]]),"%")
                           
        
            
            
        
        #print(roc_plot)
        records <- character(0)
        if(!is.na(roc$F1[[i_maxF1]]) & roc$F1[[i_maxF1]] > best_F1){
            best_F1 <- roc$F1[[i_maxF1]]
            records <- c(records,"best_F1")
        }
        
        
        tag <- paste0(tstamp,"_",epoch)
        
        for(rec in records){
            old_recs <- dir(path=logdir, pattern=paste0(rec,"_",tstamp),full.names = TRUE)
            if (any(file.exists(old_recs))){
                file.remove(old_recs)
            }
            roc_plot <- ggplot()+
                geom_line(data = roc , aes(x= FPR,y = TPR))+
                labs(title = roc_title, 
                     subtitle = roc_subtitle,
                     x = "FPR (1 - specificity)",
                     y = "TPR (sensitivity)") +
                geom_point(mapping = aes(x = roc$FPR[[i_maxF1]],
                                         y = roc$TPR[[i_maxF1]]), 
                           shape = 7, size=3, colour="red")
            roc_file <- paste0(tag,"_",rec,".csv")
            plot_file <- paste0(tag,"_",rec,".png")
            pars_file <- paste0(tag,"_",rec,".Rdata")
            write_csv(x=roc,path=file.path(logdir,roc_file))
            pars <- sess$run(parameters)
            save(pars,file=file.path(logdir,pars_file))
            png(file.path(logdir,plot_file))
            print(roc_plot)
            dev.off()
            
            
        }
            
        alpha <- 0.9*(1-exp(0.9-epoch))     
        secs_this_epoch <- Sys.time()-time
        avg_secs_per_epoch <- (1-alpha)*secs_this_epoch + alpha*secs_last_epoch
        secs_remaining <- avg_secs_per_epoch*(num_epochs-epoch)
        eta <- as.character(Sys.time()+secs_remaining)
        cat("\n\t",sprintf("%.2f",secs_this_epoch),"s/epoch. ETA: ",eta,"\n")
        
        secs_last_epoch <- secs_this_epoch
        
    }
    
    

    sess$close()
 
}
