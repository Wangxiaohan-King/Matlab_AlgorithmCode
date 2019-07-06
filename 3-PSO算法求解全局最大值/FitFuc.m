function fitness=FitFuc(x)
fitness=1/(1+(2.1*(1-x+2*x.^2).*exp(-x.^2/2)));