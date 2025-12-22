class PromptFactory:
    """Generates structured prompts for the LLM to fill DSL skeletons."""

    @staticmethod
    def get_system_prompt() -> str:
        return (
            "You are a specialized data synthesis agent for Aetheris Smart Home OS.\n"
            "Your goal is to convert technical DSL Skeletons into realistic training samples.\n\n"
            "CRITICAL RULES:\n"
            "1. VALUE ALIGNMENT: The specific values you put in the DSL (replacing placeholders) "
            "MUST be mentioned in the Natural Language (NL) variants.\n"
            "   - If DSL has 'Kitchen', NL must mention 'Kitchen'.\n"
            "   - If DSL has 'LAMP_001', NL must mention 'LAMP_001' or a very clear synonym.\n"
            "2. Replace placeholders (e.g., '{{ROOM_NAME}}') with realistic smart home values.\n"
            "3. Keep the DSL structure, operators, and single quotes EXACTLY as provided.\n"
            "4. Generate 3 diverse NL variants (Casual, Formal, Indirect).\n"
            "5. Return ONLY a valid JSON object."
        )

    @staticmethod
    def get_user_prompt(skeleton: dict) -> str:
        return (
            f"Table Name: {skeleton['table_name']}\n"
            f"Table Description: {skeleton['table_description']}\n"
            f"Columns Information: {skeleton['columns_info']}\n"
            f"DSL Skeleton: {skeleton['dsl_skeleton']}\n\n"
            "Provide the result in this JSON format:\n"
            "{\n"
            '  "final_dsl": "the dsl with values instead of placeholders",\n'
            '  "nl_variants": ["variant 1", "variant 2", "variant 3"]\n'
            "}"
        )
