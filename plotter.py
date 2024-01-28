
# importing the required module
import matplotlib.pyplot as plt

datadir = "./data/";
filename = "cost";

x, y = [], [];
idx = 0;
with open(datadir + filename) as file:
    for line in file.readlines():
        idx += 1;

        x.append(idx);
        y.append(float(line.strip()));

# plotting the points 
plt.plot(x, y)
 
# naming the x axis
plt.xlabel("time")
# naming the y axis
plt.ylabel(filename)
 
# giving a title to my graph
plt.title(filename)
 
# function to show the plot
plt.show()
