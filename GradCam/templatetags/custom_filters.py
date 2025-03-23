from django import template
import os

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """Template filter to access dictionary items by key."""
    return dictionary.get(key)

@register.filter
def basename(path):
    """Template filter to get the basename of a file path."""
    return os.path.basename(path) 