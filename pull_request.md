# How to Submit a Pull Request on GitHub

This tutorial will guide you through the process of submitting a pull request to a GitHub repository. We'll use the repository [kennykguo](https://github.com/kennykguo?tab=repositories) as an example.

---

## Step 1: Fork the Repository

1. Go to the repository you want to contribute to. For example, navigate to [kennykguo's repositories](https://github.com/kennykguo?tab=repositories).
2. Click the "Fork" button in the top-right corner of the repository page. This will create a copy of the repository under your GitHub account.

---

## Step 2: Clone Your Fork

1. After forking, go to your GitHub profile and find the forked repository.
2. Click the "Code" button and copy the URL of your forked repository.
3. Open your terminal and run the following command to clone the repository to your local machine:

   git clone https://github.com/YOUR_USERNAME/REPOSITORY_NAME.git

   Replace `YOUR_USERNAME` with your GitHub username and `REPOSITORY_NAME` with the name of the repository.

---

## Step 3: Create a New Branch

1. Navigate to the cloned repository:

   cd REPOSITORY_NAME

2. Create a new branch for your changes:

   git checkout -b your-branch-name

   Replace `your-branch-name` with a descriptive name for your branch.

---

## Step 4: Make Your Changes

1. Make the necessary changes to the code or files in the repository.
2. Save your changes.

---

## Step 5: Commit Your Changes

1. Stage your changes:

   git add .

2. Commit your changes with a descriptive message:

   git commit -m "Your commit message"

---

## Step 6: Push Your Changes to GitHub

1. Push your changes to your forked repository:

   git push origin your-branch-name

---

## Step 7: Create a Pull Request

1. Go to your forked repository on GitHub.
2. You should see a banner at the top of the repository page with a button that says "Compare & pull request." Click it.
3. On the pull request page, ensure that the base repository is the original repository you forked from, and the head repository is your forked repository.
4. Write a title and description for your pull request, explaining the changes you made.
5. Click the "Create pull request" button.

---

## Step 8: Wait for Review

1. The repository maintainers will review your pull request. They may ask for changes or merge it if everything looks good.
2. If changes are requested, make the necessary updates and push them to the same branch. The pull request will automatically update.

---

## Step 9: Celebrate!

Once your pull request is merged, congratulations! You've successfully contributed to the repository.

---

That's it! You've now learned how to submit a pull request on GitHub. Happy coding!