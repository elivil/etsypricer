# EtsyPricer - Craft the Right Price

## About the project

EtsyPricer is a web application that provides estimates of competitive pricing for handmade items made by Etsy sellers and gives recommendations for improved item descriptions. This application uses a text description and an image uploaded by the user as inputs for a machine learning model trained on historical data. The application was built in two weeks as part of the Insight Data Science program. It uses Python, Flask, AWS and SQL. The live version is available at http://etsypricer.host.

## Project motivation
Etsy is an online marketplace that helps makers sell handmade items. It has been growing at a signficant pace, amassing almost 2M sellers. Last year it has added 180,000 sellers, so almost one in ten of Etsy sellers are new to the platform. As a new seller, one faces the challenge of pricing items competitively. Etsy achieved $922.5M GMS (gross merchandise sales) in Q3 2018 from which we can estimate that almost 40M items were sold on its marketplace. Assuming that new sellers accounted for 10% of those sales, sellers underpricing by $1 would lose out on $4M in sales revenue. Such a mispricing would also have a negative impact on Etsy revenue given that Etsy charges 5% fee on each sold item.

When a new seller adds a listing to Etsy, they are prompted to enter the price:
![Enter price prompt](imgs/enterprice.png)
