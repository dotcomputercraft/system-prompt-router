"""
Command-line interface for System Prompt Router.

This module provides a comprehensive CLI for interacting with the System Prompt Router,
including interactive mode, batch processing, and configuration management.
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, List

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn

from .router import SystemPromptRouter
from .config import Config
from .library import PromptLibrary


console = Console()


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, config, verbose):
    """System Prompt Router - Intelligent prompt routing using embeddings."""
    ctx.ensure_object(dict)
    ctx.obj['config_file'] = config
    ctx.obj['verbose'] = verbose
    
    if verbose:
        console.print("[dim]Verbose mode enabled[/dim]")


@cli.command()
@click.argument('query', required=False)
@click.option('--prompt-library', '-p', type=click.Path(exists=True), 
              help='Path to prompt library file')
@click.option('--top-k', '-k', default=3, help='Number of top matches to show')
@click.option('--no-response', is_flag=True, help='Show matches without generating response')
@click.option('--method', default='cosine', 
              type=click.Choice(['cosine', 'dot_product', 'euclidean']),
              help='Similarity calculation method')
@click.option('--interactive', '-i', is_flag=True, help='Start interactive mode')
@click.pass_context
def query(ctx, query, prompt_library, top_k, no_response, method, interactive):
    """Process a query and generate response using best matching prompt."""
    try:
        # Initialize router
        config = Config(ctx.obj.get('config_file'))
        router = SystemPromptRouter(config)
        
        # Load prompt library
        if prompt_library:
            router.load_prompt_library(prompt_library)
        else:
            # Load default prompts
            default_prompts_path = Path(__file__).parent.parent.parent / "config" / "default_prompts.json"
            if default_prompts_path.exists():
                router.load_prompt_library(str(default_prompts_path))
            else:
                console.print("[red]No prompt library found. Please specify --prompt-library[/red]")
                return
        
        if interactive or not query:
            _interactive_mode(router, top_k, method, no_response)
        else:
            _process_single_query(router, query, top_k, method, no_response)
            
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        if ctx.obj.get('verbose'):
            console.print_exception()
        sys.exit(1)


def _interactive_mode(router, top_k, method, no_response):
    """Run interactive query mode."""
    console.print(Panel.fit(
        "[bold blue]System Prompt Router - Interactive Mode[/bold blue]\n"
        "Type your queries and get intelligent prompt routing.\n"
        "Commands: /help, /list, /stats, /quit",
        border_style="blue"
    ))
    
    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]Query[/bold cyan]").strip()
            
            if not user_input:
                continue
            
            if user_input.startswith('/'):
                if user_input == '/quit' or user_input == '/exit':
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                elif user_input == '/help':
                    _show_help()
                elif user_input == '/list':
                    _show_prompts(router)
                elif user_input == '/stats':
                    _show_stats(router)
                else:
                    console.print("[red]Unknown command. Type /help for available commands.[/red]")
                continue
            
            _process_single_query(router, user_input, top_k, method, no_response)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")


def _process_single_query(router, query, top_k, method, no_response):
    """Process a single query and display results."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        # Find best matches
        progress.add_task("Finding best matching prompts...", total=None)
        matches = router.find_best_prompt(query, top_k=top_k, method=method)
        
        # Display matches
        _display_matches(matches, method)
        
        if not no_response and matches:
            # Generate response
            progress.add_task("Generating response...", total=None)
            response_data = router.generate_response(query)
            _display_response(response_data)


def _display_matches(matches, method):
    """Display prompt matches in a formatted table."""
    if not matches:
        console.print("[red]No matching prompts found.[/red]")
        return
    
    table = Table(title=f"Top Matching Prompts (Method: {method})")
    table.add_column("Rank", style="cyan", width=6)
    table.add_column("Prompt Name", style="green")
    table.add_column("Similarity Score", style="yellow", width=15)
    table.add_column("Description", style="white")
    
    for i, (name, score, _) in enumerate(matches, 1):
        # Get description from router
        description = "N/A"
        try:
            prompt_details = router.get_prompt_details(name)
            if prompt_details:
                description = prompt_details.get("description", "N/A")[:50] + "..."
        except:
            pass
        
        table.add_row(
            str(i),
            name,
            f"{score:.4f}",
            description
        )
    
    console.print(table)


def _display_response(response_data):
    """Display the generated response."""
    console.print("\n" + "="*60)
    console.print(Panel(
        response_data["response"],
        title=f"[bold green]Response (using: {response_data['matched_prompt']})[/bold green]",
        border_style="green"
    ))
    
    # Show metadata
    metadata_text = f"Model: {response_data['model_used']}"
    if response_data.get('tokens_used'):
        metadata_text += f" | Tokens: {response_data['tokens_used']}"
    if response_data.get('similarity_score'):
        metadata_text += f" | Similarity: {response_data['similarity_score']:.4f}"
    
    console.print(f"[dim]{metadata_text}[/dim]")


def _show_help():
    """Show help information."""
    help_text = """
[bold]Available Commands:[/bold]

[cyan]/help[/cyan]     - Show this help message
[cyan]/list[/cyan]     - List all available prompts
[cyan]/stats[/cyan]    - Show router statistics
[cyan]/quit[/cyan]     - Exit interactive mode

[bold]Usage:[/bold]
Simply type your query and press Enter to get intelligent prompt routing and response generation.
    """
    console.print(Panel(help_text, title="Help", border_style="blue"))


def _show_prompts(router):
    """Show all available prompts."""
    prompts = router.list_prompts()
    
    if not prompts:
        console.print("[red]No prompts loaded.[/red]")
        return
    
    table = Table(title="Available Prompts")
    table.add_column("Name", style="green")
    table.add_column("Description", style="white")
    
    for name, description in prompts.items():
        table.add_row(name, description[:80] + "..." if len(description) > 80 else description)
    
    console.print(table)


def _show_stats(router):
    """Show router statistics."""
    stats = router.get_stats()
    
    # Library stats
    lib_stats = stats.get('library', {})
    console.print(f"[bold]Library:[/bold] {lib_stats.get('total_prompts', 0)} prompts loaded")
    
    # Embedding stats
    emb_stats = stats.get('embeddings', {})
    console.print(f"[bold]Embeddings:[/bold] {emb_stats.get('cached_embeddings', 0)} cached, "
                 f"{emb_stats.get('cache_file_size_mb', 0):.2f} MB cache")
    
    # Config stats
    config_stats = stats.get('config', {})
    console.print(f"[bold]Models:[/bold] {config_stats.get('embedding_model', 'N/A')} (embedding), "
                 f"{config_stats.get('openai_model', 'N/A')} (OpenAI)")


@cli.command()
@click.option('--prompt-library', '-p', type=click.Path(exists=True),
              help='Path to prompt library file')
@click.pass_context
def list_prompts(ctx, prompt_library):
    """List all available prompts."""
    try:
        config = Config(ctx.obj.get('config_file'))
        router = SystemPromptRouter(config)
        
        if prompt_library:
            router.load_prompt_library(prompt_library)
        else:
            default_prompts_path = Path(__file__).parent.parent.parent / "config" / "default_prompts.json"
            if default_prompts_path.exists():
                router.load_prompt_library(str(default_prompts_path))
        
        _show_prompts(router)
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@click.option('--prompt-library', '-p', type=click.Path(exists=True),
              help='Path to prompt library file')
@click.option('--format', default='json', type=click.Choice(['json', 'csv', 'txt']),
              help='Output format')
@click.pass_context
def batch(ctx, input_file, output, prompt_library, format):
    """Process multiple queries from a file."""
    try:
        config = Config(ctx.obj.get('config_file'))
        router = SystemPromptRouter(config)
        
        # Load prompt library
        if prompt_library:
            router.load_prompt_library(prompt_library)
        else:
            default_prompts_path = Path(__file__).parent.parent.parent / "config" / "default_prompts.json"
            if default_prompts_path.exists():
                router.load_prompt_library(str(default_prompts_path))
        
        # Read input queries
        with open(input_file, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]
        
        console.print(f"Processing {len(queries)} queries...")
        
        results = []
        with Progress() as progress:
            task = progress.add_task("Processing queries...", total=len(queries))
            
            for query in queries:
                try:
                    response_data = router.generate_response(query)
                    results.append(response_data)
                except Exception as e:
                    results.append({"query": query, "error": str(e)})
                
                progress.advance(task)
        
        # Save results
        if output:
            _save_batch_results(results, output, format)
            console.print(f"[green]Results saved to {output}[/green]")
        else:
            _display_batch_results(results)
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)


def _save_batch_results(results, output_file, format):
    """Save batch processing results to file."""
    if format == 'json':
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    elif format == 'csv':
        import csv
        with open(output_file, 'w', newline='') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
    elif format == 'txt':
        with open(output_file, 'w') as f:
            for result in results:
                f.write(f"Query: {result.get('user_query', 'N/A')}\n")
                f.write(f"Response: {result.get('response', 'N/A')}\n")
                f.write(f"Prompt: {result.get('matched_prompt', 'N/A')}\n")
                f.write("-" * 60 + "\n")


def _display_batch_results(results):
    """Display batch processing results."""
    for i, result in enumerate(results, 1):
        console.print(f"\n[bold cyan]Query {i}:[/bold cyan] {result.get('user_query', 'N/A')}")
        if 'error' in result:
            console.print(f"[red]Error: {result['error']}[/red]")
        else:
            console.print(f"[green]Response:[/green] {result.get('response', 'N/A')[:100]}...")
            console.print(f"[yellow]Prompt:[/yellow] {result.get('matched_prompt', 'N/A')}")


@cli.command()
@click.pass_context
def validate(ctx):
    """Validate router setup and configuration."""
    try:
        config = Config(ctx.obj.get('config_file'))
        router = SystemPromptRouter(config)
        
        # Load default prompts for validation
        default_prompts_path = Path(__file__).parent.parent.parent / "config" / "default_prompts.json"
        if default_prompts_path.exists():
            router.load_prompt_library(str(default_prompts_path))
        
        errors = router.validate_setup()
        
        if not errors:
            console.print("[green]✓ Router setup is valid![/green]")
        else:
            console.print("[red]Validation errors found:[/red]")
            for error in errors:
                console.print(f"  • {error}")
            sys.exit(1)
        
    except Exception as e:
        console.print(f"[red]Validation failed: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--prompt-library', '-p', type=click.Path(exists=True),
              help='Path to prompt library file')
@click.pass_context
def stats(ctx, prompt_library):
    """Show detailed router statistics."""
    try:
        config = Config(ctx.obj.get('config_file'))
        router = SystemPromptRouter(config)
        
        if prompt_library:
            router.load_prompt_library(prompt_library)
        else:
            default_prompts_path = Path(__file__).parent.parent.parent / "config" / "default_prompts.json"
            if default_prompts_path.exists():
                router.load_prompt_library(str(default_prompts_path))
        
        stats = router.get_stats()
        
        # Display detailed stats
        console.print(Panel.fit(
            json.dumps(stats, indent=2),
            title="Router Statistics",
            border_style="blue"
        ))
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)


if __name__ == '__main__':
    cli()

