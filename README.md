# House-Rent
House rent prediction
**Problem Statement:**
In the real estate industry, determining the appropriate rental price for a property is crucial for
property owners, tenants, and property management companies. Accurate rent predictions can
help landlords set competitive prices, tenants make informed rental decisions, and property
management companies optimize their portfolio management.
The goal of this project is to develop a data-driven model that predicts the rental price of
residential properties based on relevant features. By analyzing historical rental data and
property attributes, the model aims to provide accurate and reliable rent predictions.

This dataset has lots of features that includes the address, amenities of the house, property age, property size, floor,water supply, facing building type, parking and rent..

This dataset has been preprocessed in cutting down features, removing duplicates,typecasting, handling missing values and finally created necessary features..

Using these features several analysis like correlation analysis, geospatial analysis,and analysis of the impact that the different features on the target i.e rent.

Finally several models have been used to get best output.. Since Random Forest had the best of all, it was optimised by hypertuning...

The optimised model had the leaset mae and mse error and r2 score of 70..

This model was saved using pickle module and developed an app using streamlit.

