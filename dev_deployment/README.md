# Starting a Dev Environment on Modal

Optional: Generate an SSH key pair:

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
# or
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
# enter "~/.ssh/id_rsa_modal" as the saved file name

# add key to terminal
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa_modal
```

Need to create a volume for persistent storage:
(Only need to run once)

```bash
modal volume create trace-bench-dev
```

Run it as a blocking command:

```
modal run dev_modal_image.py
```

Run it detached (might need to shut down app through the Modal interface):

```
modal run --detach dev_modal_image.py
```

Or simply run it in a screen or tmux session.