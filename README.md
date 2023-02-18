# Rainbow: Combining Improvements in Deep Reinforcement Learning, Hessel et Al.
# An algorithmic trading application

From Hessel's paper, we adapt the rainbow and compare the algorithms that compose it in a trading framework. We use daily returns on CAC40 stocks between 2020 and today. (2020-2021 for train and 2022-today for test).
It appears that the rainbow outperforms the simple buy and hold strategy, while ensuring an attractive level of risk compared to the other more volatile algorithms.

## Participants
Marc Veron

Samy Mdihi

Timothe Guillaume-Li

## Installation
Download as zip file locally

## Usage
Run main.py : it will load the pre-trained models (on 10 stocks from 01/01/2020 to 30/12/2021) and will test on 5 CAC40 stocks : Bouygues, Unibail, Renault, Thales and Eurofins.

Configure the training of each algo in its dedicated folder and run the file. Once trained, recover the pth file in the output folder and place it at the root of the algo folder, so that it is recognized by the main.
