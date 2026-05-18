"""Shared pytest configuration for coherax tests."""

import jax

jax.config.update("jax_enable_x64", True)
