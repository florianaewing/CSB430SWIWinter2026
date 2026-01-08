# CSB430 SW&I

This repository contains the demo code covered in class.

## Weekly Structure

- Each week includes both a **solution** folder and a **starter code** folder.
- If you’d like to follow along during lecture, **pull down the starter code** for that week.
- I will be updating this every week with new Demo material to match the pace of the course.

## How to Pull the Code

1. **Clone the repository once** (the first time you set it up):
   ```bash
   git clone <repository-url>
   cd AD320WebDevLectureCodeFall2025
   ```
2. **At the start of each class, pull the latest updates**:
   ```bash
   git pull
   ```
   This will fetch the newest lecture code.

## Working With Your Own Copy

To avoid overwriting your personal changes when new code is pushed, use one of these approaches:

### Option 1: Create a Branch for Your Work

1. Create a new branch for your changes:
   ```bash
   git checkout -b my-work
   ```
2. Do your work on this branch.
3. When you need the latest class updates:
   ```bash
   git checkout main
   git pull
   git checkout my-work
   git merge main
   ```
   Your branch will now include the newest lecture code without losing your changes.

### Option 2: Fork the Repository

1. On GitHub, click **Fork** to create a copy in your own account.
2. Clone your fork:
   ```bash
   git clone <your-fork-url>
   ```
3. Work freely in your fork.
4. If you want to fetch updates from the class repository, add it as a remote:
   ```bash
   git remote add upstream <class-repo-url>
   git fetch upstream
   git merge upstream/main
   ```
   Either approach protects your work while letting you keep up with new lecture code.
