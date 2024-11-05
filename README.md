# Julia Copilot

## How to Run the Tool

**TODO:** make this paragraph cleaner later

We cloned and cleaned (removed any non julia files) from the all the repos
marked with Julia as a language from github. We wanted to add the cached zip to
the repo, but Github doesn't allow us to do that because the file is too big
(350M c.a.).

What you can do, if you don't want to go through the process of running
`clone.py`, you can simply download the result, which is present of the server.

You can execute the following line (note, this line is assuming your are in the
root folder of the repo):

```bash
scp <YOUR_USER>@gym.si.usi.ch:/home/SA24-G3/project2/data/repos.zip data/
```
