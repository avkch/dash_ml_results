# dash_ml_results
## This repository contains python script for creating a dashboard for visual examination and analysis of binary machine learning results.
Requires Pandas and Plotly/Dash packages
To start the script, open a console and go to the folder containing *ml_results_test.py* file. Type: `python ml_results_test.py`
Open in your browser http://127.0.0.1:8050/ and you will see the dashboard.
Example file *sample_file.xlsx* provided in the repository contains example results from spam filter. *Predicted_values* column contains scores calculated by ML algorithm with values between 1 and -1.  The results closer to 1 are more likely to be ham and results closer to -1 spam. 
![Dash ML result Demo](dash_use.gif)
