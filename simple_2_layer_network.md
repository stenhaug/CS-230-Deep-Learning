simple\_2\_layer\_network
================

# Summary

X: features \* observations  
W1: neurons in layer i \* feature-weights  
bi: neurons in layer i \* 1 (need broadcasting)  
Z1 = Wi \* X + bi: neurons in layer i \* observations  
A1 = activation\_function(Zi): neurons in layer i \* observations

Wi — (neurons in layer i \* neurons in layer i - 1 )  
x  
X or Ai — (neurons in layer i - 1 \* observations)  
\=  
Zi — (neurons in layer i \* observations)

# Implement example

Let’s implement a 2 x 4 x 1 NN:

![copyright coursera](images/2_layer_network.png)

What each of the activation functions looks like:

``` r
tibble(x = seq(-4, 4, 0.01)) %>% 
   mutate(
      tanh = tanh(x),
      sigmoid = sigmoid(x)
   ) %>%
   gather(var, y, -x) %>% 
   ggplot(aes(x, y, color = var)) +
   geom_point()
```

![](simple_2_layer_network_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->

## Hidden layer

X is our input matrix. Each column is one of our 5 observations. The row
is a feature.

``` r
(X <- matrix(c(1, 2), nrow = 2, ncol = 5))
```

    ##      [,1] [,2] [,3] [,4] [,5]
    ## [1,]    1    1    1    1    1
    ## [2,]    2    2    2    2    2

4 neurons in the 1st layer. Dimensions of are to \* from which is 4 \*
2. This is to \* from.

``` r
(W1 <- matrix(c(1, 2), nrow = 4, ncol = 2, byrow = TRUE))
```

    ##      [,1] [,2]
    ## [1,]    1    2
    ## [2,]    1    2
    ## [3,]    1    2
    ## [4,]    1    2

We also have b which is like the intercept of that layer. Its dimensions
are to \* 1

``` r
(b1 <- matrix(1, nrow = 4, ncol = 1))
```

    ##      [,1]
    ## [1,]    1
    ## [2,]    1
    ## [3,]    1
    ## [4,]    1

Now this is where we are after the first hidden layer

``` r
(Z1 <- W1 %*% X + broadcast(b1))
```

    ##      [,1] [,2] [,3] [,4] [,5]
    ## [1,]    6    6    6    6    6
    ## [2,]    6    6    6    6    6
    ## [3,]    6    6    6    6    6
    ## [4,]    6    6    6    6    6

We apply the activation function to Z.

``` r
(A1 <- tanh(Z1))
```

    ##           [,1]      [,2]      [,3]      [,4]      [,5]
    ## [1,] 0.9999877 0.9999877 0.9999877 0.9999877 0.9999877
    ## [2,] 0.9999877 0.9999877 0.9999877 0.9999877 0.9999877
    ## [3,] 0.9999877 0.9999877 0.9999877 0.9999877 0.9999877
    ## [4,] 0.9999877 0.9999877 0.9999877 0.9999877 0.9999877

## Output layer

Must have 1 layer so we have a singular prediction.

``` r
(W2 <- matrix(1, nrow = 1, ncol = 4, byrow = TRUE))
```

    ##      [,1] [,2] [,3] [,4]
    ## [1,]    1    1    1    1

``` r
(b2 <- matrix(-0.5, nrow = 1, ncol = 1))
```

    ##      [,1]
    ## [1,] -0.5

``` r
(Z2 <- (W2 %*% A1 + broadcast(b2)))
```

    ##          [,1]     [,2]     [,3]     [,4]     [,5]
    ## [1,] 3.499951 3.499951 3.499951 3.499951 3.499951

``` r
(A2 <- sigmoid(Z2))
```

    ##           [,1]      [,2]      [,3]      [,4]      [,5]
    ## [1,] 0.9706864 0.9706864 0.9706864 0.9706864 0.9706864
