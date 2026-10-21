// chat_template.h - LFM2 / LFM2.5 chat template (the GGUF "tokenizer.chat_template", ported natively; no jinja engine needed).
//
//   <|startoftext|>[<|im_start|>system\n{system + "\nList of tools: [..]"}<|im_end|>\n]
//   (<|im_start|>{role}\n{content}<|im_end|>\n)*  [<|im_start|>assistant\n]
//
// Past assistant turns lose their reasoning: if the content contains "</think>", only the text after the last "</think>" (trimmed)
// is kept - except for the last assistant message, or everything when keep_past_thinking is set.
#pragma once
#include "json.h"
#include <string>

namespace ferro {

struct ChatTemplateOptions {
    bool add_generation_prompt = true;
    bool keep_past_thinking = false;
};

// messages: JSON array of {"role", "content"} (content: string, or an array of {"type":"text","text":..} parts which are
// joined with "\n", as llama.cpp's server does); tools: optional JSON array (each rendered with the jinja `tojson` filter). Throws std::invalid_argument.
std::string lfm_chat_render(const Json& messages, const Json* tools, const ChatTemplateOptions& opt = {});

}  // namespace ferro
