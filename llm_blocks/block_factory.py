
from llm_blocks import blocks


block_factories = {}


def register_block_type(type_name):
    def decorator(factory):
        block_factories[type_name] = factory
        return factory

    return decorator


@register_block_type("block")
def create_block(*args, **kwargs):
    return blocks.Block(
        config=blocks.OpenAIConfig(*args, **kwargs),
        message_handler=blocks.MessageHandler(system_message=kwargs.get("system_message")),
    )


@register_block_type("template")
def create_template_block(template, *args, **kwargs):
    return blocks.TemplateBlock(
        template,
        config=blocks.OpenAIConfig(*args, **kwargs),
        message_handler=blocks.MessageHandler(system_message=kwargs.get("system_message")),
    )


@register_block_type("chat")
def create_chat_block(*args, **kwargs):
    return blocks.ChatBlock(
        config=blocks.OpenAIConfig(*args, **kwargs),
        message_handler=blocks.MessageHandler(system_message=kwargs.get("system_message")),
    )


def block_factory(type, *args, **kwargs):
    factory = block_factories.get(type)
    if factory is None:
        raise ValueError(f"Unknown block type: {type}")
    return factory(*args, **kwargs)
