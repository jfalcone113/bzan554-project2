Help complete the group assignment 2. For the most part I know whats going on, but need assistance with the code. When responding to a prompt, be sure to provide the class relevant answer and if requested, quickly explain what the code does (only break it down a lot if requested). The instructions for group assignment 2 are pasted below, followed by the group assignment 1 instructions. If needed, refer to my class folder containing all my class materials located at Desktop/Backpack/bzan554 directory. Also if needed, Ballings' link to these class materials is available at https://www.dropbox.com/scl/fo/c7k9wuzo0opy63l4nf7eh/AHiVr2RoAHP3TdfNsaD8RaQ?rlkey=m7cwjkb0zrsxd5xz11xqmdd1m&e=1&dl=0.

Group Assignment 2
Tuning Deep Neural Networks on real-life Business Data

Background and Goal:
The goal of this assignment is to gather experience on the sensitivity of the algorithm to different kinds of tuning parameters.

As you will see, there is no code submission for this assignment; only a presentation is due. The goal of this assignment is to reward you for doing the work and, most importantly, for learning from it.

The presentation itself carries relatively little weight toward your final grade (typically 5%; see the syllabus). This is intentional. If your only goal were to complete the assignment, you could probably do so in an hour by relying heavily on tools such as Codex or Claude Code. But you would learn very little from doing that.

Instead, think of the assignment as preparation for Test 2. The test will include questions related to this assignment and may require you to write and reason through code using pen and paper.
So, while you are welcome to use AI tools as part of your learning process, make sure that you understand the code, the decisions you made, and why the code works. That understanding, not simply producing working code, is the real objective of the assignment.

Data:
All data is contained in pricing.csv (same data as assignment 1). The data is copyrighted and
confidential. One line in the data represents a product selling on the company’s e-commerce website. If a product goes out-of-stock and then becomes in-stock again a new line is created in the data for that product

• sku: stock keeping unit
• price: the price of the product on the website
• quantity: total quantity sold
• order: identifies the nth time the product was in-stock and selling on the website
• duration: how long the product appeared on the site before going out-of-stock
• category: product category

Note 1: All categorical variables are integer encoded (not necessarily consecutively). All numeric variables are divided by a constant.

Variables

Input variables
• sku
• price
• order
• duration
• category

Response variable:
• quantity

Tuning parameters
Batch size, number of hidden layers, number hidden neurons, hidden activation functions (sigmoid, tanh, relu, leaky relu, prelu, elu), optimizers (plain SGD, momentum, nesterov, adagrad, rmsprop, adam, learning rate scheduling), sequence models, …, anything we have seen in class.

Deliverables:
Presentation (pdf or PowerPoint)
-Useful insights on the sensitivity to tuning parameters
-Details on training and testing
-Statement of who did what to show contributions of all team members

Deadline:
See syllabus.

Submission link:
https://www.dropbox.com/request/fnbc8zhd3ubefgqvkai3

Groups
See groups_from_Zaretzki.xlsx

Group Assignment 1

Incremental Learning on Large Data

Learning objective:
Estimating a feed-forward neural network on data that is larger than what can be held in a typical computer’s RAM.

As you will see, there is no code submission for this assignment; only a presentation is due. The goal of this assignment is to reward you for doing the work and, most importantly, for learning from it.

The presentation itself carries relatively little weight toward your final grade (typically 5%; see the syllabus). This is intentional. If your only goal were to complete the assignment, you could probably do so in an hour by relying heavily on tools such as Codex or Claude Code. But you would learn very little from doing that.

Instead, think of the assignment as preparation for Test 1. The test will include questions related to this assignment and may require you to write and reason through code using pen and paper.
So, while you are welcome to use AI tools as part of your learning process, make sure that you understand the code, the decisions you made, and why the code works. That understanding, not simply producing working code, is the real objective of the assignment.

Data:
All data is contained in pricing.csv. The data is copyrighted and confidential. One line in the data represents a product selling on the company’s e-commerce website. If a product goes out-of-stock and then becomes in-stock again a new line is created in the data for that product

• sku: stock keeping unit
• price: the price of the product on the website
• quantity: total quantity sold
• order: identifies the nth time the product was in-stock and selling on the website
• duration: how long the product appeared on the site before going out-of-stock
• category: product category

Note 1: All categorical variables are integer encoded (not necessarily consecutively). All numeric variables are divided by a constant.

Variables

Input variables
• sku
• price
• order
• duration
• category

Response variable:
• quantity

Deliverables:
Your goal is to build a feed forward neural network that predicts quantity sold. The neural network should have three hidden layers, with sigmoid activation. The model should be learned incrementally: read in one record (or a mini-batch), update the model, repeat. The only deliverable is a presentation (PDF or PowerPoint). There is no code submission. Ensure
your correct group ID is in the file name and on the first slide.

In your presentation include the following:
- The R2, on a test set (which you create) and training set (do not use your test set for
training or tuning): ∑(𝑦"− 𝑦#))𝑅! = 1 −∑(𝑦"− 𝑦+)!
- A learning curve: a plot with on the x-axis the number of instances learned, and on the y-
axis the moving average of the MSE.
- A plot showing variable importances
- Multiple partial dependence plots
- Information on RAM usage
- Information on training time
- Statement of who did what to show contributions of all team members

Deadline:
See syllabus.
Submission link:
https://www.dropbox.com/request/ogczi51z6x1ls3xeunum
Groups
See groups_from_Zaretzki.xlsx

