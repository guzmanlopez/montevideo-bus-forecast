import typer


def msg_download(message: str = ""):
    typer.secho("âŦī¸ Downloading data...", fg=typer.colors.BRIGHT_CYAN, bold=True)
    typer.secho(f"{message}", fg=typer.colors.CYAN)


def msg_write(message: str = ""):
    typer.secho("đž Writing data...", fg=typer.colors.BRIGHT_MAGENTA, bold=True)
    typer.secho(f"{message}", fg=typer.colors.MAGENTA)


def msg_done(message: str = "Done!"):
    typer.secho(f"â {message} \n", fg=typer.colors.BRIGHT_GREEN, bold=True)


def msg_load(message: str = ""):
    typer.secho("đī¸ââī¸ Loading data...", fg=typer.colors.BRIGHT_WHITE, bold=True)
    typer.secho(f"{message}", fg=typer.colors.WHITE)


def msg_process(message: str = ""):
    typer.secho("âī¸ Processing...", fg=typer.colors.BRIGHT_BLUE, bold=True)
    typer.secho(f"{message}", fg=typer.colors.BRIGHT_BLUE)


def msg_bus(bus_line: str = ""):
    typer.secho(f" đ {bus_line}", fg=typer.colors.BRIGHT_WHITE, bold=True)


def msg_bus_stop(message: str = ""):
    typer.secho(f"đ {message}", fg=typer.colors.BRIGHT_WHITE, bold=True)


def msg_bus_track(message: str = ""):
    typer.secho(f"đŖī¸ {message}", fg=typer.colors.BRIGHT_WHITE, bold=True)


def msg_warn(message: str = ""):
    typer.secho(f"â ī¸ {message}", fg=typer.colors.BRIGHT_YELLOW, bold=True)


def msg_info(message: str = ""):
    typer.secho(f"âšī¸ {message}", fg=typer.colors.BRIGHT_CYAN, bold=False)
