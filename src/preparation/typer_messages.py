import typer


def msg_download(message: str):
    typer.secho(f"⬇️ {message}", fg=typer.colors.CYAN)


def msg_writte(message: str):
    typer.secho(f"✏️ {message}", fg=typer.colors.MAGENTA)


def msg_done(message: str):
    typer.secho(f"✅ {message}", fg=typer.colors.GREEN)
