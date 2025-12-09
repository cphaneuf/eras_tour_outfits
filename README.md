# Machine Learning Analysis of Taylor Swift's Outfits During the Eras Tour

## RESEARCH QUESTIONS

- <b>RQ #1a:</b> how well can we predict the Lover Bodysuit (i.e., the 1st outfit of the show) from day, city, country, region, and night?
- <b>RQ #1b:</b> how well can we predict the Lover Bodysuit (i.e., the 1st outfit of the show) from outfit delays AND day, city, country, region, and night?
- <b>RQ #2a:</b> how well can we predict the Fearless Dress (i.e., the 2nd outfit of the show) from the Lover outfit AND outfit delays AND day, city, country, region, and night?
- <b>RQ #2b:</b> how well can we predict the Red Shirt (i.e., the 3rd outfit of the show) from the Lover outfit AND the Fearless outfit AND outfit delays AND day, city, country, region, and night? (forthcoming)

(cont.)

## DATA SOURCES

- <b>Training/testing data:</b> https://docs.google.com/spreadsheets/d/1WZyhckHAwOosHGA65h5dp5SHL5aoUMiyYXcY1k5MYUM/edit?gid=174092590#gid=174092590
- <b>Benchmarking data:</b> https://docs.google.com/spreadsheets/d/1uvVEEqZsUbWCb61vSmpJRGfwtwoxbU70hkKgJiramHQ/edit?gid=1500331289#gid=1500331289

## MODELS AND TECHNIQUES

- Distribution visualization
- Multinomial logistic regression
- Penalization
- Parameter tuning (forthcoming)
- Cross validation (forthcoming)
- Feature ranking (forthcoming)
- Benchmarking (forthcoming)

## HOUSEKEEPING

<b>To summarize the contents of this repo:</b>

<i>data.csv</i> is a database of Eras Tour outfits

<i>code.py</i> performs data visualizations and conducts the analyses to address the research questions, above

<i>usa_regions_ref.png</i> is the reference used for the labeling the Region column of data.csv

<i>base_env.yaml</i> recreates my environment in Anaconda-Navigator

The code was developed using the Spyder IDE (version 5.4.3), launched through Anaconda-Navigator. To run, be sure to change the value of <i>abs_path</i> to reflect the absolute path to this repo on your personal machine.

## CONTACT

Questions can be directed to me: Camille Phaneuf-Hadd (cphaneuf@g.harvard.edu).
