import typer


def msg_download(message: str = ""):
    typer.secho("⬇️ Downloading data...", fg=typer.colors.BRIGHT_CYAN, bold=True)
    typer.secho(f"{message}", fg=typer.colors.CYAN)


def msg_write(message: str = ""):
    typer.secho("✏ Writing data...", fg=typer.colors.BRIGHT_MAGENTA, bold=True)
    typer.secho(f"{message}", fg=typer.colors.MAGENTA)


def msg_done(message: str = "Done!"):
    typer.secho(f"✅ {message} \n", fg=typer.colors.BRIGHT_GREEN)


def msg_load(message: str = ""):
    typer.secho("🏋️‍♀️ Loading data...", fg=typer.colors.BRIGHT_YELLOW, bold=True)
    typer.secho(f"{message}", fg=typer.colors.YELLOW)


def msg_process(message: str = ""):
    typer.secho("⚙️ Processing...", fg=typer.colors.BRIGHT_YELLOW, bold=True)
    typer.secho(f"{message}", fg=typer.colors.YELLOW)


def msg_bus(bus_line: str = ""):
    typer.secho(f"🚌 {bus_line}", fg=typer.colors.BRIGHT_YELLOW, bold=True)
