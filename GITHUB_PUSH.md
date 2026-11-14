# Push after extracting the repository

Open a terminal inside the extracted `brain-mri-sae-study` directory. Create an
empty GitHub repository in your own account, then run these commands, replacing
YOUR_USERNAME and YOUR_REPOSITORY. Git must be installed and authenticated.

```bash
git init
git add .
git status --short
git commit -m "Add MRI SAE study code and aggregate results"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Inspect the staged files before committing. Keep data and weights outside version
control. If Git author identity is missing, configure your own name/email, not a
fabricated identity. No repository creation, commit or push was performed here.
