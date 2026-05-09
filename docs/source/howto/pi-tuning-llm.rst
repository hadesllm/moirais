Tuning a Raspberry Pi 5 for LLM inference
==========================================

Runtime kernel and filesystem tunings that meaningfully improve
local-LLM responsiveness on a Pi 5 hosting Ollama. All reversible.
No custom kernel build required.

The baseline
------------

A Pi 5 with 16 GB RAM running Raspberry Pi OS (Debian bookworm,
kernel 6.12.x) with Ollama serving Gemma 4 2B (~9.6 GB, Q4_K_M) will
work out of the box, but you'll notice:

- Swap fills up within a day, even with plenty of free RAM.
- NVMe I/O stalls under mixed interactive + LLM load.
- :code:`sysctl` isn't in :code:`PATH` for non-root users.
- :code:`TERM=dumb` when you SSH in from some clients.
- Boot console noise interrupts SSH sessions.

Each of these has a one-file fix.

Installed files
---------------

.. code-block:: text

    /etc/sysctl.d/99-moirais-luci.conf          # runtime kernel tuning
    /etc/udev/rules.d/00-nvme-bfq.rules      # I/O scheduler per device
    ~/.bashrc.d/00-moirais-luci.sh              # shell env, PATH, prompt
    ~/.tmux.conf                              # modern tmux defaults
    ~/.inputrc                                # readline quality-of-life

All reversible by deleting the respective file.

sysctl tunings
--------------

:code:`/etc/sysctl.d/99-moirais-luci.conf` (apply with :code:`sudo sysctl
--system`):

.. code-block:: ini

    # Swap behavior: don't page out an in-use ~9 GB model to reclaim
    # ~200 MB of disk cache. Default 60 is too aggressive.
    vm.swappiness = 10

    # Keep filesystem metadata resident (loop.sh re-reads the same files
    # every iteration).
    vm.vfs_cache_pressure = 50

    # OOM killer headroom — 128 MB reserve.
    vm.min_free_kbytes = 131072

    # Network buffers for local HTTP (Ollama) and SSH multiplex.
    net.core.rmem_default = 262144
    net.core.rmem_max = 16777216
    net.core.wmem_default = 262144
    net.core.wmem_max = 16777216
    net.ipv4.tcp_rmem = 4096 87380 16777216
    net.ipv4.tcp_wmem = 4096 65536 16777216

    # Faster connection reuse for short-lived Ollama requests.
    net.ipv4.tcp_fin_timeout = 15

    # Suppress info-level kernel messages on terminals (was 4 4 1 7).
    kernel.printk = 3 4 1 3

    # More PID headroom and file handles — tmux + systemd user + Ollama
    # can accumulate.
    kernel.pid_max = 4194304
    fs.file-max = 2097152
    fs.inotify.max_user_watches = 524288
    fs.inotify.max_user_instances = 1024

**Measure before and after.** Watch :code:`free -h` and :code:`grep ^Swap
/proc/meminfo` over 24 hours. On MOIRAIS's Pi, swappiness=10 cut average
swap usage from ~1.5 GB to ~0.4 GB.

I/O scheduler per device
------------------------

:code:`/etc/udev/rules.d/00-nvme-bfq.rules`:

.. code-block:: udev

    # NVMe: BFQ for mixed interactive + background load
    ACTION=="add|change", KERNEL=="nvme[0-9]n[0-9]", ATTR{queue/scheduler}="bfq"

    # SD card: mq-deadline, which handles small-block erase better
    ACTION=="add|change", KERNEL=="mmcblk[0-9]", ATTR{queue/scheduler}="mq-deadline"
    ACTION=="add|change", KERNEL=="sd[a-z]", ATTR{queue/scheduler}="mq-deadline"

Apply:

.. code-block:: bash

    sudo cp 00-nvme-bfq.rules /etc/udev/rules.d/
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    cat /sys/block/nvme0n1/queue/scheduler   # should show [bfq]

**Why BFQ on NVMe but not SD.** BFQ's fairness guarantees help when
Ollama is reading 9 GB of model weights while tmux panes are
scrolling and systemd is writing journal entries. On SD cards, BFQ's
overhead hurts more than its fairness helps, because SD I/O is
dominated by flash erase/program cycles, not queue depth.

Shell environment
-----------------

:code:`~/.bashrc.d/00-moirais-luci.sh` (sourced via a loader in
:code:`.bashrc`):

.. code-block:: bash

    # Only interactive shells
    case $- in *i*) ;; *) return ;; esac

    # Upgrade TERM=dumb (some SSH clients default to this) when on a TTY
    if [ -t 1 ] && { [ "$TERM" = "dumb" ] || [ -z "$TERM" ]; }; then
        export TERM=xterm-256color
    fi

    # Add /sbin + /usr/sbin so sysctl, ip, ss are bare-name reachable
    case ":$PATH:" in *":/sbin:"*) ;; *) export PATH="$PATH:/sbin:/usr/sbin" ;; esac

    # Larger, de-duped history with timestamps
    export HISTSIZE=50000 HISTFILESIZE=200000
    export HISTCONTROL=ignoreboth:erasedups
    export HISTTIMEFORMAT='%F %T  '
    shopt -s histappend cmdhist checkwinsize

    # Color prompt with git branch + non-zero exit indicator
    _gb() { git symbolic-ref --short HEAD 2>/dev/null | sed 's/^/ (/;s/$/)/'; }
    _ex() { local ec=$?; [ "$ec" -eq 0 ] || printf '\[\e[31m\][%d]\[\e[0m\] ' "$ec"; }
    PS1='\[\e[36m\]\u@\h\[\e[0m\]:\[\e[33m\]\w\[\e[35m\]$(_gb)\[\e[0m\] $(_ex)\$ '

    # Aliases that matter for an LLM-host Pi
    alias ll='ls -lah --color=auto'
    alias pi-temp='vcgencmd measure_temp 2>/dev/null'
    alias pi-throttled='vcgencmd get_throttled 2>/dev/null'
    alias luci-status='systemctl --user status luci-loop.service --no-pager'
    alias luci-tail='tail -f ~/universe/journal/LATEST.md'

Loader snippet at the end of :code:`.bashrc`:

.. code-block:: bash

    if [ -d "$HOME/.bashrc.d" ]; then
        for f in "$HOME/.bashrc.d/"*.sh; do
            [ -r "$f" ] && . "$f"
        done
        unset f
    fi

tmux and readline
-----------------

:code:`~/.tmux.conf`:

.. code-block:: tmux

    set -g default-terminal "tmux-256color"
    set -ga terminal-overrides ",xterm-256color:Tc,tmux-256color:Tc"
    set -g mouse on
    set -g history-limit 100000
    set -g base-index 1
    setw -g pane-base-index 1
    set -g renumber-windows on
    setw -g mode-keys vi

    bind r source-file ~/.tmux.conf \; display "reloaded"
    bind | split-window -h -c "#{pane_current_path}"
    bind - split-window -v -c "#{pane_current_path}"
    bind h select-pane -L
    bind j select-pane -D
    bind k select-pane -U
    bind l select-pane -R

:code:`~/.inputrc`:

.. code-block:: text

    set completion-ignore-case on
    set show-all-if-ambiguous on
    set bell-style none
    set colored-stats on
    set colored-completion-prefix on
    "\e[A": history-search-backward
    "\e[B": history-search-forward

The prefix-based history search (Up/Down) is the single biggest
quality-of-life change for long SSH sessions. Type :code:`git
` then Up/Down cycles through only your past :code:`git` commands,
not every command in history.

Optional: zram + modern CLI tools
---------------------------------

.. code-block:: bash

    sudo apt install -y zram-tools fzf bat ripgrep eza btop
    sudo systemctl enable --now zramswap.service

:code:`zram` compresses memory pages instead of swapping to disk.
On a Pi 5 with 16 GB RAM and a 9 GB model, it rarely kicks in — but
when it does, it's orders of magnitude faster than SD swap.

Install :code:`fzf`, :code:`bat`, :code:`ripgrep`, :code:`eza`, and
:code:`btop` if you SSH into the Pi often. They change nothing about
performance but change everything about ergonomics.

What this guide does NOT do
---------------------------

- **Does not touch :code:`/boot/firmware/config.txt`** — that affects
  GPU memory split, overclocking, and would require a reboot. You
  should decide what goes there.
- **Does not change the CPU governor.** The default :code:`ondemand`
  governor is correct for Pi 5 under mixed load.
- **Does not rebuild any kernel modules.** Custom kernel work
  requires a spare partition and 2-3 hours of build time; it's
  scoped separately.

Verification
------------

After applying everything, :code:`ssh` in fresh and check:

.. code-block:: bash

    # TERM upgraded
    echo $TERM                               # xterm-256color

    # sysctl applied
    sysctl vm.swappiness                     # vm.swappiness = 10
    sysctl vm.vfs_cache_pressure              # vm.vfs_cache_pressure = 50

    # scheduler correct
    cat /sys/block/nvme0n1/queue/scheduler   # ...[bfq]...
    cat /sys/block/sda/queue/scheduler        # ...[mq-deadline]...

    # prompt looks colorful
    # (if PS1 shows the git branch after you cd into a repo, you're good)

Trade-offs
----------

- **sysctl changes are system-wide.** If the Pi hosts other workloads
  (web services, home automation), tune with their needs in mind too.
- **BFQ has higher CPU overhead than mq-deadline.** Negligible on Pi
  5, but worth knowing if you port this to a Pi 4.
- **Modern CLI tools add ~200 MB of disk.** Nothing compared to a
  9 GB model, but if you're on a 32 GB SD card, budget for it.
