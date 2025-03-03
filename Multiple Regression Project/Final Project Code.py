import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics as st
import thinkstats2
import thinkplot
from sklearn import linear_model
import statsmodels.api as sm


# Importing the Dataset
led = pd.read_csv("Life Expectancy Data2.csv")

# Creating the Histograms
plt.hist(led["Status"], bins = 3)
plt.xlabel("Status")
plt.show()
plt.hist(led["Alcohol"], bins = 13, color = "red")
plt.xlabel("Consumption per captia in liters")
plt.show()
plt.hist(led["GDP"], bins = 6, color = "darkgreen")
plt.xlabel("GDP per capita")
plt.show()
plt.hist(led["HDI"], bins = 10, color = "darkblue")
plt.xlabel("HDI")
plt.show()
plt.hist(led["Total_expenditure"], bins = 10, color = "purple")
plt.xlabel("% of Health Expenditure vs Total")
plt.show()
plt.hist(led["Life_expectancy"], bins = 10, color = "pink")
plt.xlabel("Life expectancy in years")
plt.show()

# Generating Summary Statstics
print("Summary Statistics For Alcohol")
print("------------------------------")
print(led["Alcohol"].describe())
print("mode", "       ", st.mode(led["Alcohol"]))

print("Summary Statistics For GDP")
print("------------------------------")
print(led["GDP"].describe())
print("mode", "     ", st.mode(led["GDP"]))

print("Summary Statistics For HDI")
print("------------------------------")
print(led["HDI"].describe())
print("mode", "       ", st.mode(led["HDI"]))

print("Summary Statistics For Total Expenditure")
print("------------------------------")
print(led["Total_expenditure"].describe())
print("mode", "       ", st.mode(led["Total_expenditure"]))

print("Summary Statistics For Life Expectancy")
print("------------------------------")
print(led["Life_expectancy"].describe())
print("mode", "       ", st.mode(led["Life_expectancy"]))

# Checking for outliers by z-score
outliers=[]
def detect_outlier(data_1):
    
    threshold=3
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)
    
    
    for y in data_1:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers

outliers_alcohol = detect_outlier(led["Alcohol"])
outliers=[]
outliers_GDP = detect_outlier(led["GDP"])
outliers=[]
outliers_HDI = detect_outlier(led["HDI"])
outliers=[]
outliers_TE = detect_outlier(led["Total_expenditure"])
outliers=[]
outliers_LE = detect_outlier(led["Life_expectancy"])

print("Outliers for Alcohol:", outliers_alcohol)
print("Outliers for GDP:", outliers_GDP)
print("Outliers for HDI:", outliers_HDI)
print("Outliers for Total expenditure:", outliers_TE)
print("Outliers for Life Expectancy:", outliers_LE)

# Splitting the Data by Status and Looking at Life Expectancy
developing = led[led.Status == "Developing"]
developed = led[led.Status != "Developing"]
developing_pmf = thinkstats2.Pmf(developing.Life_expectancy, label='developing')
developed_pmf = thinkstats2.Pmf(developed.Life_expectancy, label='developed')
thinkplot.Hist(developing_pmf)
thinkplot.Config(xlabel='Developing', ylabel='Pmf')
thinkplot.Hist(developed_pmf)
thinkplot.Config(xlabel='Developed', ylabel='Pmf')
thinkplot.Show()

# CDF of Life Expectancy
LE_cdf = thinkstats2.Cdf(led.Life_expectancy)
thinkplot.Cdf(LE_cdf)
thinkplot.Show(xlabel='Life Expectancy', ylabel='Cdf')

# Creating the Normal Probability Plot
expectancies = led.Life_expectancy.dropna()
mean, var = thinkstats2.TrimmedMeanVar(expectancies, p=0.01)
std = np.sqrt(var)

xs = [-4, 4]
fxs, fys = thinkstats2.FitLine(xs, mean, std)
thinkplot.Plot(fxs, fys, linewidth=4, color='0.8')

xs, ys = thinkstats2.NormalProbability(expectancies)
thinkplot.Plot(xs, ys, label='Life Expectancy')

thinkplot.Config(title='Normal probability plot',
                 xlabel='Standard deviations from mean',
                 ylabel='Life Expectancy (Years)')
plt.show()

# Creating a new dataset for non-zero HDI
led_HDI = led[led.HDI != 0]
## This assumes that the 0 values in HDI are actually supposed to be NA values.

# Creating the scatter plots for each variable vs Life_expectancy
plt.scatter(led.Alcohol, led.Life_expectancy, color = "red")
plt.xlabel("Alcohol Consumption")
plt.ylabel("Life Expectancy")
plt.title("Life Expectancy vs. Alcohol Consumption")
plt.show()

plt.scatter(led_HDI.HDI, led_HDI.Life_expectancy, color = "darkblue")
plt.xlabel("HDI")
plt.ylabel("Life Expectancy")
plt.title("Life Expectancy vs. Human Development Index")
plt.show()

plt.scatter(led.Total_expenditure, led.Life_expectancy, color = "purple")
plt.xlabel("Total Expenditure")
plt.ylabel("Life Expectancy")
plt.title("Life Expectancy vs. Total Expenditure")
plt.show()

plt.scatter(led.Life_expectancy, led.GDP, color = "darkgreen")
plt.xlabel("Life Expectancy")
plt.ylabel("GDP")
plt.title("GDP vs. Life Expectancy")
plt.show()

plt.scatter(led.Life_expectancy, np.log(led.GDP), color = "darkgreen")
plt.xlabel("Life Expectancy")
plt.ylabel("Log(GDP)")
plt.title("Log(GDP) vs. Life Expectancy")
plt.show()

# Running the regression analysis using all 5 variables vs Life Expectancy 
## Note: ledr is the same data set but it contains a recode for the Status variable,
### where 1 = Developed, and 0 = Developing
ledr = pd.read_csv("Life Expectancy Data3.csv")
led_nna = ledr.dropna()
led_nna["LogGDP"] = np.log(led_nna.GDP)
X = led_nna[['Status', 'Alcohol', 'HDI', 'Total_expenditure', 'LogGDP']]
Y = led_nna['Life_expectancy']

reg = linear_model.LinearRegression()
reg.fit(X,Y)
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
print(model.summary())

