# Filename: app.py
"""
Deploy Model CivitAI ke Modal.com dengan FastAPI
Features: Text-to-Image, Image-to-Image, ControlNet, Multi-LoRA
Fixed: All bugs resolved, consistent API key handling, GPU L4
"""

import modal
from pathlib import Path
import io
import base64

# ===================================================================
# KONFIGURASI
# ===================================================================
app = modal.App("civitai-api-fastapi")

DEFAULT_NEGATIVE_PROMPT = (
    "(worst quality, low quality, normal quality, blurry, fuzzy, pixelated), "
    "(extra limbs, extra fingers, malformed hands, missing fingers, extra digit, "
    "fused fingers, too many hands, bad hands, bad anatomy), "
    "(ugly, deformed, disfigured), "
    "(text, watermark, logo, signature), "
    "(worst quality, low quality, normal quality:1.4), (jpeg artifacts, blurry, grainy), ugly, duplicate, morbid, mutilated, (deformed, disfigured), (bad anatomy, bad proportions), (extra limbs, extra fingers, fused fingers, too many fingers, long neck), (mutated hands, bad hands, poorly drawn hands), (missing arms, missing legs), malformed limbs, (cross-eyed, bad eyes, asymmetrical eyes), (cleavage), signature, watermark, username, text, error, "
    "out of frame, out of focus, "
    "cropped, close-up, portrait, headshot, medium shot, upper body, bust shot, face, out of frame"
)

DEFAULT_POSITIVE_PROMPT_SUFFIX = (
    "masterpiece, best quality, 8k, photorealistic, intricate details, wide shot, "
    "(full body shot)"
)

# ===================================================================
# KONFIGURASI LORA & CONTROLNET
# ===================================================================
LORA_DIR = "/loras"
LORA_MODELS = {
  #  "// A. Essential Tools & Helpers (15 LoRAs)": {},
    "tool_sdxl_offset_noise": {"url": "https://civitai.com/api/download/models/135931", "filename": "tool_sdxl_offset_noise.safetensors"},
    "tool_add_detail_xl": {"url": "https://civitai.com/api/download/models/223332", "filename": "tool_add_detail_xl.safetensors"},
    "tool_perfect_hands_xl": {"url": "https://civitai.com/api/download/models/209359", "filename": "tool_perfect_hands_xl.safetensors"},
    "tool_perfect_eyes_xl": {"url": "https://civitai.com/api/download/models/225330", "filename": "tool_perfect_eyes_xl.safetensors"},
    "tool_better_faces_xl": {"url": "https://civitai.com/api/download/models/141443", "filename": "tool_better_faces_xl.safetensors"},
    "tool_skin_texture_xl": {"url": "https://civitai.com/api/download/models/174129", "filename": "tool_skin_texture_xl.safetensors"},
    "tool_depth_of_field_xl": {"url": "https://civitai.com/api/download/models/125439", "filename": "tool_depth_of_field_xl.safetensors"},
    "tool_lens_flare_xl": {"url": "https://civitai.com/api/download/models/150123", "filename": "tool_lens_flare_xl.safetensors"},
    "tool_light_and_shadow_xl": {"url": "https://civitai.com/api/download/models/209315", "filename": "tool_light_and_shadow_xl.safetensors"},
    "tool_hair_detailer_xl": {"url": "https://civitai.com/api/download/models/185363", "filename": "tool_hair_detailer_xl.safetensors"},
    "tool_fabric_textures_xl": {"url": "https://civitai.com/api/download/models/131175", "filename": "tool_fabric_textures_xl.safetensors"},
    "tool_composition_control_xl": {"url": "https://civitai.com/api/download/models/142917", "filename": "tool_composition_control_xl.safetensors"},
    "tool_god_rays_xl": {"url": "https://civitai.com/api/download/models/121769", "filename": "tool_god_rays_xl.safetensors"},
    "tool_wet_surfaces_xl": {"url": "https://civitai.com/api/download/models/133185", "filename": "tool_wet_surfaces_xl.safetensors"},
    "tool_film_slide_border_xl": {"url": "https://civitai.com/api/download/models/128362", "filename": "tool_film_slide_border_xl.safetensors"},

  #  "// B. Photographic & Cinematic Styles (18 LoRAs)": {},
    "style_cinematic_lighting_xl": {"url": "https://civitai.com/api/download/models/152433", "filename": "style_cinematic_lighting_xl.safetensors"},
    "style_film_grain_fuji_xl": {"url": "https://civitai.com/api/download/models/132841", "filename": "style_film_grain_fuji_xl.safetensors"},
    "style_film_grain_kodak_xl": {"url": "https://civitai.com/api/download/models/134937", "filename": "style_film_grain_kodak_xl.safetensors"},
    "style_vintage_photo_xl": {"url": "https://civitai.com/api/download/models/131558", "filename": "style_vintage_photo_xl.safetensors"},
    "style_neon_noir_xl": {"url": "https://civitai.com/api/download/models/126222", "filename": "style_neon_noir_xl.safetensors"},
    "style_low_light_photography_xl": {"url": "https://civitai.com/api/download/models/203498", "filename": "style_low_light_photography_xl.safetensors"},
    "style_hasselblad_look_xl": {"url": "https://civitai.com/api/download/models/183181", "filename": "style_hasselblad_look_xl.safetensors"},
    "style_motion_picture_film_xl": {"url": "https://civitai.com/api/download/models/175389", "filename": "style_motion_picture_film_xl.safetensors"},
    "style_vibrant_colors_xl": {"url": "https://civitai.com/api/download/models/129759", "filename": "style_vibrant_colors_xl.safetensors"},
    "style_infrared_photo_xl": {"url": "https://civitai.com/api/download/models/158778", "filename": "style_infrared_photo_xl.safetensors"},
    "style_dramatic_portraits_xl": {"url": "https://civitai.com/api/download/models/147386", "filename": "style_dramatic_portraits_xl.safetensors"},
    "style_cinestill_800t_xl": {"url": "https://civitai.com/api/download/models/126023", "filename": "style_cinestill_800t_xl.safetensors"},
    "style_macro_photography_xl": {"url": "https://civitai.com/api/download/models/120251", "filename": "style_macro_photography_xl.safetensors"},
    "style_polaroid_photo_xl": {"url": "https://civitai.com/api/download/models/127196", "filename": "style_polaroid_photo_xl.safetensors"},
    "style_lomography_xl": {"url": "https://civitai.com/api/download/models/126922", "filename": "style_lomography_xl.safetensors"},
    "style_soft_glow_and_bloom_xl": {"url": "https://civitai.com/api/download/models/124748", "filename": "style_soft_glow_and_bloom_xl.safetensors"},
    "style_split_toning_xl": {"url": "https://civitai.com/api/download/models/144215", "filename": "style_split_toning_xl.safetensors"},
    "style_bleach_bypass_film_xl": {"url": "https://civitai.com/api/download/models/121549", "filename": "style_bleach_bypass_film_xl.safetensors"},

   # "// C. Artistic & Illustration Styles (17 LoRAs)": {},
    "style_oil_painting_xl": {"url": "https://civitai.com/api/download/models/126322", "filename": "style_oil_painting_xl.safetensors"},
    "style_watercolor_painting_xl": {"url": "https://civitai.com/api/download/models/130142", "filename": "style_watercolor_painting_xl.safetensors"},
    "style_ink_wash_painting_sumi-e_xl": {"url": "https://civitai.com/api/download/models/123274", "filename": "style_ink_wash_painting_sumi-e_xl.safetensors"},
    "style_pencil_sketch_xl": {"url": "https://civitai.com/api/download/models/152914", "filename": "style_pencil_sketch_xl.safetensors"},
    "style_tarot_card_xl": {"url": "https://civitai.com/api/download/models/127117", "filename": "style_tarot_card_xl.safetensors"},
    "style_art_deco_xl": {"url": "https://civitai.com/api/download/models/141029", "filename": "style_art_deco_xl.safetensors"},
    "style_vaporwave_xl": {"url": "https://civitai.com/api/download/models/121342", "filename": "style_vaporwave_xl.safetensors"},
    "style_pop_art_xl": {"url": "https://civitai.com/api/download/models/127192", "filename": "style_pop_art_xl.safetensors"},
    "style_pixel_art_xl": {"url": "https://civitai.com/api/download/models/120096", "filename": "style_pixel_art_xl.safetensors"},
    "style_3d_render_character_xl": {"url": "https://civitai.com/api/download/models/125307", "filename": "style_3d_render_character_xl.safetensors"},
    "style_childrens_storybook_xl": {"url": "https://civitai.com/api/download/models/143527", "filename": "style_childrens_storybook_xl.safetensors"},
    "style_horror_manga_junji_ito_xl": {"url": "https://civitai.com/api/download/models/121543", "filename": "style_horror_manga_junji_ito_xl.safetensors"},
    "style_alphonse_mucha_xl": {"url": "https://civitai.com/api/download/models/121118", "filename": "style_alphonse_mucha_xl.safetensors"},
    "style_van_gogh_xl": {"url": "https://civitai.com/api/download/models/131700", "filename": "style_van_gogh_xl.safetensors"},
    "style_ukiyo-e_xl": {"url": "https://civitai.com/api/download/models/121900", "filename": "style_ukiyo-e_xl.safetensors"},
    "style_acrylic_painting_xl": {"url": "https://civitai.com/api/download/models/132810", "filename": "style_acrylic_painting_xl.safetensors"},
    "style_cyber_sigilism_xl": {"url": "https://civitai.com/api/download/models/123389", "filename": "style_cyber_sigilism_xl.safetensors"},

  #  "// D. Anime & Manga Styles (12 LoRAs)": {},
    "style_anime_ghibli_xl": {"url": "https://civitai.com/api/download/models/152640", "filename": "style_anime_ghibli_xl.safetensors"},
    "style_anime_makoto_shinkai_xl": {"url": "https://civitai.com/api/download/models/121833", "filename": "style_anime_makoto_shinkai_xl.safetensors"},
    "style_anime_retro_80s_xl": {"url": "https://civitai.com/api/download/models/125184", "filename": "style_anime_retro_80s_xl.safetensors"},
    "style_anime_screencap_xl": {"url": "https://civitai.com/api/download/models/148425", "filename": "style_anime_screencap_xl.safetensors"},
    "style_webtoon_korea_xl": {"url": "https://civitai.com/api/download/models/132561", "filename": "style_webtoon_korea_xl.safetensors"},
    "style_manga_monochrome_xl": {"url": "https://civitai.com/api/download/models/124239", "filename": "style_manga_monochrome_xl.safetensors"},
    "style_anime_cel_shader_xl": {"url": "https://civitai.com/api/download/models/132338", "filename": "style_anime_cel_shader_xl.safetensors"},
    "style_anime_fantasy_art_xl": {"url": "https://civitai.com/api/download/models/121406", "filename": "style_anime_fantasy_art_xl.safetensors"},
    "style_anime_cute_chibi_xl": {"url": "https://civitai.com/api/download/models/121087", "filename": "style_anime_cute_chibi_xl.safetensors"},
    "style_anime_gacha_splash_xl": {"url": "https://civitai.com/api/download/models/129188", "filename": "style_anime_gacha_splash_xl.safetensors"},
    "style_kpop_photoshoot_xl": {"url": "https://civitai.com/api/download/models/165561", "filename": "style_kpop_photoshoot_xl.safetensors"},
    "style_90s_kpop_look_xl": {"url": "https://civitai.com/api/download/models/171565", "filename": "style_90s_kpop_look_xl.safetensors"},
    
   # "// E. Characters (Games, Anime, Other) (31 LoRAs)": {},
    "char_tifa_lockhart_ff7_xl": {"url": "https://civitai.com/api/download/models/120391", "filename": "char_tifa_lockhart_ff7_xl.safetensors"},
    "char_2b_nier_automata_xl": {"url": "https://civitai.com/api/download/models/120485", "filename": "char_2b_nier_automata_xl.safetensors"},
    "char_raiden_shogun_genshin_xl": {"url": "https://civitai.com/api/download/models/123490", "filename": "char_raiden_shogun_genshin_xl.safetensors"},
    "char_ganyu_genshin_xl": {"url": "https://civitai.com/api/download/models/125791", "filename": "char_ganyu_genshin_xl.safetensors"},
    "char_furina_genshin_xl": {"url": "https://civitai.com/api/download/models/203277", "filename": "char_furina_genshin_xl.safetensors"},
    "char_kafka_honkai_star_rail_xl": {"url": "https://civitai.com/api/download/models/120613", "filename": "char_kafka_honkai_star_rail_xl.safetensors"},
    "char_seele_honkai_star_rail_xl": {"url": "https://civitai.com/api/download/models/120937", "filename": "char_seele_honkai_star_rail_xl.safetensors"},
    "char_acheron_honkai_star_rail_xl": {"url": "https://civitai.com/api/download/models/300481", "filename": "char_acheron_honkai_star_rail_xl.safetensors"},
    "char_astarion_baldurs_gate_3_xl": {"url": "https://civitai.com/api/download/models/145885", "filename": "char_astarion_baldurs_gate_3_xl.safetensors"},
    "char_shadowheart_baldurs_gate_3_xl": {"url": "https://civitai.com/api/download/models/146191", "filename": "char_shadowheart_baldurs_gate_3_xl.safetensors"},
    "char_link_zelda_xl": {"url": "https://civitai.com/api/download/models/120963", "filename": "char_link_zelda_xl.safetensors"},
    "char_zelda_totk_xl": {"url": "https://civitai.com/api/download/models/120448", "filename": "char_zelda_totk_xl.safetensors"},
    "char_jinx_arcane_xl": {"url": "https://civitai.com/api/download/models/120536", "filename": "char_jinx_arcane_xl.safetensors"},
    "char_yor_forger_spy_family_xl": {"url": "https://civitai.com/api/download/models/122137", "filename": "char_yor_forger_spy_family_xl.safetensors"},
    "char_makima_csm_xl": {"url": "https://civitai.com/api/download/models/142171", "filename": "char_makima_csm_xl.safetensors"},
    "char_gojo_satoru_jjk_xl": {"url": "https://civitai.com/api/download/models/122144", "filename": "char_gojo_satoru_jjk_xl.safetensors"},
    "char_asuna_sao_xl": {"url": "https://civitai.com/api/download/models/157540", "filename": "char_asuna_sao_xl.safetensors"},
    "char_rem_re_zero_xl": {"url": "https://civitai.com/api/download/models/122049", "filename": "char_rem_re_zero_xl.safetensors"},
    "char_cyberpunk_edgerunners_lucy_xl": {"url": "https://civitai.com/api/download/models/120800", "filename": "char_cyberpunk_edgerunners_lucy_xl.safetensors"},
    "char_chainsaw_man_power_xl": {"url": "https://civitai.com/api/download/models/121715", "filename": "char_chainsaw_man_power_xl.safetensors"},
    "char_taylor_swift_xl": {"url": "https://civitai.com/api/download/models/133003", "filename": "char_taylor_swift_xl.safetensors"},
    "char_ana_de_armas_xl": {"url": "https://civitai.com/api/download/models/127116", "filename": "char_ana_de_armas_xl.safetensors"},
    "char_scarlett_johansson_xl": {"url": "https://civitai.com/api/download/models/261971", "filename": "char_scarlett_johansson_xl.safetensors"},
    "char_henry_cavill_xl": {"url": "https://civitai.com/api/download/models/125028", "filename": "char_henry_cavill_xl.safetensors"},
    "char_face_russian_beauty_xl": {"url": "https://civitai.com/api/download/models/188045", "filename": "char_face_russian_beauty_xl.safetensors"},
    "char_face_asian_men_xl": {"url": "https://civitai.com/api/download/models/203480", "filename": "char_face_asian_men_xl.safetensors"},
    "char_face_asian_girl_xl": {"url": "https://civitai.com/api/download/models/126416", "filename": "char_face_asian_girl_xl.safetensors"},
    "char_face_elf_xl": {"url": "https://civitai.com/api/download/models/121703", "filename": "char_face_elf_xl.safetensors"},
    "char_face_orc_xl": {"url": "https://civitai.com/api/download/models/124233", "filename": "char_face_orc_xl.safetensors"},
    "char_face_vampire_xl": {"url": "https://civitai.com/api/download/models/121650", "filename": "char_face_vampire_xl.safetensors"},
    "char_face_general_k-idol_xl": {"url": "https://civitai.com/api/download/models/144383", "filename": "char_face_general_k-idol_xl.safetensors"},

  #  "// F. Clothing & Fashion (17 LoRAs)": {},
    "fashion_gothic_lolita_xl": {"url": "https://civitai.com/api/download/models/121774", "filename": "fashion_gothic_lolita_xl.safetensors"},
    "fashion_techwear_cyberpunk_xl": {"url": "https://civitai.com/api/download/models/120993", "filename": "fashion_techwear_cyberpunk_xl.safetensors"},
    "fashion_traditional_kimono_xl": {"url": "https://civitai.com/api/download/models/124706", "filename": "fashion_traditional_kimono_xl.safetensors"},
    "fashion_traditional_hanfu_xl": {"url": "https://civitai.com/api/download/models/121683", "filename": "fashion_traditional_hanfu_xl.safetensors"},
    "fashion_latex_xl": {"url": "https://civitai.com/api/download/models/122119", "filename": "fashion_latex_xl.safetensors"},
    "fashion_sci-fi_armor_xl": {"url": "https://civitai.com/api/download/models/125338", "filename": "fashion_sci-fi_armor_xl.safetensors"},
    "fashion_hoodie_xl": {"url": "https://civitai.com/api/download/models/120412", "filename": "fashion_hoodie_xl.safetensors"},
    "fashion_oversized_clothes_xl": {"url": "https://civitai.com/api/download/models/255091", "filename": "fashion_oversized_clothes_xl.safetensors"},
    "fashion_swimsuit_bikini_xl": {"url": "https://civitai.com/api/download/models/122709", "filename": "fashion_swimsuit_bikini_xl.safetensors"},
    "fashion_wedding_dress_xl": {"url": "https://civitai.com/api/download/models/121738", "filename": "fashion_wedding_dress_xl.safetensors"},
    "fashion_streetwear_xl": {"url": "https://civitai.com/api/download/models/124016", "filename": "fashion_streetwear_xl.safetensors"},
    "fashion_steampunk_xl": {"url": "https://civitai.com/api/download/models/121102", "filename": "fashion_steampunk_xl.safetensors"},
    "fashion_victorian_era_xl": {"url": "https://civitai.com/api/download/models/123164", "filename": "fashion_victorian_era_xl.safetensors"},
    "fashion_formal_suit_xl": {"url": "https://civitai.com/api/download/models/125774", "filename": "fashion_formal_suit_xl.safetensors"},
    "fashion_biker_jacket_xl": {"url": "https://civitai.com/api/download/models/151197", "filename": "fashion_biker_jacket_xl.safetensors"},
    "fashion_kpop_idol_stage_outfit_xl": {"url": "https://civitai.com/api/download/models/121287", "filename": "fashion_kpop_idol_stage_outfit_xl.safetensors"},
    "fashion_school_uniform_xl": {"url": "https://civitai.com/api/download/models/121223", "filename": "fashion_school_uniform_xl.safetensors"},

   # "// G. Concepts, Objects, & Poses (15 LoRAs)": {},
    "concept_curvy_chubby_girl_xl": {"url": "https://civitai.com/api/download/models/131765", "filename": "concept_curvy_chubby_girl_xl.safetensors"},
    "concept_muscular_body_xl": {"url": "https://civitai.com/api/download/models/128873", "filename": "concept_muscular_body_xl.safetensors"},
    "concept_mecha_girl_fusion_xl": {"url": "https://civitai.com/api/download/models/158762", "filename": "concept_mecha_girl_fusion_xl.safetensors"},
    "concept_dynamic_poses_xl": {"url": "https://civitai.com/api/download/models/121296", "filename": "concept_dynamic_poses_xl.safetensors"},
    "object_katana_xl": {"url": "https://civitai.com/api/download/models/124483", "filename": "object_katana_xl.safetensors"},
    "object_gundam_mecha_xl": {"url": "https://civitai.com/api/download/models/120539", "filename": "object_gundam_mecha_xl.safetensors"},
    "concept_explosion_and_smoke_xl": {"url": "https://civitai.com/api/download/models/120935", "filename": "concept_explosion_and_smoke_xl.safetensors"},
    "concept_fire_and_flames_xl": {"url": "https://civitai.com/api/download/models/130198", "filename": "concept_fire_and_flames_xl.safetensors"},
    "concept_glitch_effect_xl": {"url": "https://civitai.com/api/download/models/120930", "filename": "concept_glitch_effect_xl.safetensors"},
    "concept_crying_tears_xl": {"url": "https://civitai.com/api/download/models/126038", "filename": "concept_crying_tears_xl.safetensors"},
    "concept_reading_book_pose_xl": {"url": "https://civitai.com/api/download/models/145892", "filename": "concept_reading_book_pose_xl.safetensors"},
    "concept_hologram_xl": {"url": "https://civitai.com/api/download/models/121098", "filename": "concept_hologram_xl.safetensors"},
    "concept_floating_lanterns_xl": {"url": "https://civitai.com/api/download/models/152887", "filename": "concept_floating_lanterns_xl.safetensors"},
    "concept_robotic_arms_xl": {"url": "https://civitai.com/api/download/models/120863", "filename": "concept_robotic_arms_xl.safetensors"},
    "concept_detailed_tattoos_xl": {"url": "https://civitai.com/api/download/models/125359", "filename": "concept_detailed_tattoos_xl.safetensors"},

  #  "// H. Environments & Architecture (15 LoRAs)": {},
    "env_sci-fi_cityscape_xl": {"url": "https://civitai.com/api/download/models/121822", "filename": "env_sci-fi_cityscape_xl.safetensors"},
    "env_fantasy_landscape_xl": {"url": "https://civitai.com/api/download/models/120999", "filename": "env_fantasy_landscape_xl.safetensors"},
    "env_detailed_interiors_xl": {"url": "https://civitai.com/api/download/models/132717", "filename": "env_detailed_interiors_xl.safetensors"},
    "env_cyberpunk_street_xl": {"url": "https://civitai.com/api/download/models/121043", "filename": "env_cyberpunk_street_xl.safetensors"},
    "env_forest_woodland_xl": {"url": "https://civitai.com/api/download/models/122000", "filename": "env_forest_woodland_xl.safetensors"},
    "env_cozy_cafe_interior_xl": {"url": "https://civitai.com/api/download/models/121226", "filename": "env_cozy_cafe_interior_xl.safetensors"},
    "env_magic_forest_xl": {"url": "https://civitai.com/api/download/models/121898", "filename": "env_magic_forest_xl.safetensors"},
    "env_rainy_city_street_xl": {"url": "https://civitai.com/api/download/models/121609", "filename": "env_rainy_city_street_xl.safetensors"},
    "env_library_interior_xl": {"url": "https://civitai.com/api/download/models/121401", "filename": "env_library_interior_xl.safetensors"},
    "env_throne_room_xl": {"url": "https://civitai.com/api/download/models/121334", "filename": "env_throne_room_xl.safetensors"},
    "env_post-apocalyptic_ruins_xl": {"url": "https://civitai.com/api/download/models/120944", "filename": "env_post-apocalyptic_ruins_xl.safetensors"},
    "env_tropical_beach_xl": {"url": "https://civitai.com/api/download/models/122415", "filename": "env_tropical_beach_xl.safetensors"},
    "env_mountain_peak_xl": {"url": "https://civitai.com/api/download/models/121896", "filename": "env_mountain_peak_xl.safetensors"},
    "env_gothic_cathedral_xl": {"url": "https://civitai.com/api/download/models/121350", "filename": "env_gothic_cathedral_xl.safetensors"},
    "env_classroom_anime_xl": {"url": "https://civitai.com/api/download/models/121873", "filename": "env_classroom_anime_xl.safetensors"},

   # "// I. Vehicles & Machinery (5 LoRAs)": {},
    "vehicle_cyberpunk_car_xl": {"url": "https://civitai.com/api/download/models/121171", "filename": "vehicle_cyberpunk_car_xl.safetensors"},
    "vehicle_classic_motorcycle_xl": {"url": "https://civitai.com/api/download/models/151475", "filename": "vehicle_classic_motorcycle_xl.safetensors"},
    "vehicle_spaceship_cockpit_xl": {"url": "https://civitai.com/api/download/models/121518", "filename": "vehicle_spaceship_cockpit_xl.safetensors"},
    "vehicle_steampunk_airship_xl": {"url": "https://civitai.com/api/download/models/121565", "filename": "vehicle_steampunk_airship_xl.safetensors"},
    "vehicle_f1_race_car_xl": {"url": "https://civitai.com/api/download/models/147545", "filename": "vehicle_f1_race_car_xl.safetensors"},

   # "// K. K-Pop Idols (57 LoRAs)": {},
    "kpop_blackpink_jennie_xl": {"url": "https://civitai.com/api/download/models/153188", "filename": "kpop_blackpink_jennie_xl.safetensors"},
    "kpop_blackpink_jisoo_xl": {"url": "https://civitai.com/api/download/models/121305", "filename": "kpop_blackpink_jisoo_xl.safetensors"},
    "kpop_blackpink_rose_xl": {"url": "https://civitai.com/api/download/models/122442", "filename": "kpop_blackpink_rose_xl.safetensors"},
    "kpop_blackpink_lisa_xl": {"url": "https://civitai.com/api/download/models/121568", "filename": "kpop_blackpink_lisa_xl.safetensors"},
    "kpop_twice_sana_xl": {"url": "https://civitai.com/api/download/models/121855", "filename": "kpop_twice_sana_xl.safetensors"},
    "kpop_twice_nayeon_xl": {"url": "https://civitai.com/api/download/models/121881", "filename": "kpop_twice_nayeon_xl.safetensors"},
    "kpop_twice_momo_xl": {"url": "https://civitai.com/api/download/models/122176", "filename": "kpop_twice_momo_xl.safetensors"},
    "kpop_twice_jihyo_xl": {"url": "https://civitai.com/api/download/models/126601", "filename": "kpop_twice_jihyo_xl.safetensors"},
    "kpop_twice_tzuyu_xl": {"url": "https://civitai.com/api/download/models/121865", "filename": "kpop_twice_tzuyu_xl.safetensors"},
    "kpop_newjeans_haerin_xl": {"url": "https://civitai.com/api/download/models/141837", "filename": "kpop_newjeans_haerin_xl.safetensors"},
    "kpop_newjeans_minji_xl": {"url": "https://civitai.com/api/download/models/124317", "filename": "kpop_newjeans_minji_xl.safetensors"},
    "kpop_newjeans_hanni_xl": {"url": "https://civitai.com/api/download/models/123984", "filename": "kpop_newjeans_hanni_xl.safetensors"},
    "kpop_newjeans_danielle_xl": {"url": "https://civitai.com/api/download/models/125345", "filename": "kpop_newjeans_danielle_xl.safetensors"},
    "kpop_ive_wonyoung_xl": {"url": "https://civitai.com/api/download/models/132588", "filename": "kpop_ive_wonyoung_xl.safetensors"},
    "kpop_aespa_karina_xl": {"url": "https://civitai.com/api/download/models/119932", "filename": "kpop_aespa_karina_xl.safetensors"},
    "kpop_aespa_winter_xl": {"url": "https://civitai.com/api/download/models/120281", "filename": "kpop_aespa_winter_xl.safetensors"},
    "kpop_aespa_ningning_xl": {"url": "https://civitai.com/api/download/models/124749", "filename": "kpop_aespa_ningning_xl.safetensors"},
    "kpop_le_sserafim_chaewon_xl": {"url": "https://civitai.com/api/download/models/122046", "filename": "kpop_le_sserafim_chaewon_xl.safetensors"},
    "kpop_le_sserafim_sakura_xl": {"url": "https://civitai.com/api/download/models/121971", "filename": "kpop_le_sserafim_sakura_xl.safetensors"},
    "kpop_le_sserafim_yunjin_xl": {"url": "https://civitai.com/api/download/models/125199", "filename": "kpop_le_sserafim_yunjin_xl.safetensors"},
    "kpop_le_sserafim_kazuha_xl": {"url": "https://civitai.com/api/download/models/122045", "filename": "kpop_le_sserafim_kazuha_xl.safetensors"},
    "kpop_gidle_miyeon_xl": {"url": "https://civitai.com/api/download/models/121870", "filename": "kpop_gidle_miyeon_xl.safetensors"},
    "kpop_gidle_minnie_xl": {"url": "https://civitai.com/api/download/models/122061", "filename": "kpop_gidle_minnie_xl.safetensors"},
    "kpop_gidle_yuqi_xl": {"url": "https://civitai.com/api/download/models/121852", "filename": "kpop_gidle_yuqi_xl.safetensors"},
    "kpop_itzy_yeji_xl": {"url": "https://civitai.com/api/download/models/122146", "filename": "kpop_itzy_yeji_xl.safetensors"},
    "kpop_itzy_ryujin_xl": {"url": "https://civitai.com/api/download/models/121849", "filename": "kpop_itzy_ryujin_xl.safetensors"},
    "kpop_itzy_yuna_xl": {"url": "https://civitai.com/api/download/models/121845", "filename": "kpop_itzy_yuna_xl.safetensors"},
    "kpop_red_velvet_irene_xl": {"url": "https://civitai.com/api/download/models/121303", "filename": "kpop_red_velvet_irene_xl.safetensors"},
    "kpop_red_velvet_seulgi_xl": {"url": "https://civitai.com/api/download/models/121415", "filename": "kpop_red_velvet_seulgi_xl.safetensors"},
    "kpop_iu_soloist_xl": {"url": "https://civitai.com/api/download/models/121302", "filename": "kpop_iu_soloist_xl.safetensors"},
    "kpop_bts_jungkook_xl": {"url": "https://civitai.com/api/download/models/129731", "filename": "kpop_bts_jungkook_xl.safetensors"},
    "kpop_bts_v_taehyung_xl": {"url": "https://civitai.com/api/download/models/120932", "filename": "kpop_bts_v_taehyung_xl.safetensors"},
    "kpop_bts_jimin_xl": {"url": "https://civitai.com/api/download/models/120933", "filename": "kpop_bts_jimin_xl.safetensors"},
    "kpop_bts_rm_xl": {"url": "https://civitai.com/api/download/models/261908", "filename": "kpop_bts_rm_xl.safetensors"},
    "kpop_bts_jin_xl": {"url": "https://civitai.com/api/download/models/121319", "filename": "kpop_bts_jin_xl.safetensors"},
    "kpop_bts_suga_xl": {"url": "https://civitai.com/api/download/models/121320", "filename": "kpop_bts_suga_xl.safetensors"},
    "kpop_bts_jhope_xl": {"url": "https://civitai.com/api/download/models/121321", "filename": "kpop_bts_jhope_xl.safetensors"},
    "kpop_stray_kids_felix_xl": {"url": "https://civitai.com/api/download/models/137021", "filename": "kpop_stray_kids_felix_xl.safetensors"},
    "kpop_stray_kids_hyunjin_xl": {"url": "https://civitai.com/api/download/models/121309", "filename": "kpop_stray_kids_hyunjin_xl.safetensors"},
    "kpop_stray_kids_bang_chan_xl": {"url": "https://civitai.com/api/download/models/122153", "filename": "kpop_stray_kids_bang_chan_xl.safetensors"},
    "kpop_stray_kids_lee_know_xl": {"url": "https://civitai.com/api/download/models/151740", "filename": "kpop_stray_kids_lee_know_xl.safetensors"},
    "kpop_stray_kids_han_xl": {"url": "https://civitai.com/api/download/models/151187", "filename": "kpop_stray_kids_han_xl.safetensors"},
    "kpop_seventeen_mingyu_xl": {"url": "https://civitai.com/api/download/models/122105", "filename": "kpop_seventeen_mingyu_xl.safetensors"},
    "kpop_seventeen_wonwoo_xl": {"url": "https://civitai.com/api/download/models/121404", "filename": "kpop_seventeen_wonwoo_xl.safetensors"},
    "kpop_seventeen_jeonghan_xl": {"url": "https://civitai.com/api/download/models/121407", "filename": "kpop_seventeen_jeonghan_xl.safetensors"},
    "kpop_txt_yeonjun_xl": {"url": "https://civitai.com/api/download/models/121304", "filename": "kpop_txt_yeonjun_xl.safetensors"},
    "kpop_txt_soobin_xl": {"url": "https://civitai.com/api/download/models/122143", "filename": "kpop_txt_soobin_xl.safetensors"},
    "kpop_txt_beomgyu_xl": {"url": "https://civitai.com/api/download/models/121510", "filename": "kpop_txt_beomgyu_xl.safetensors"},
    "kpop_enhypen_sunghoon_xl": {"url": "https://civitai.com/api/download/models/122154", "filename": "kpop_enhypen_sunghoon_xl.safetensors"},
    "kpop_enhypen_heeseung_xl": {"url": "https://civitai.com/api/download/models/140026", "filename": "kpop_enhypen_heeseung_xl.safetensors"},
    "kpop_enhypen_sunoo_xl": {"url": "https://civitai.com/api/download/models/137020", "filename": "kpop_enhypen_sunoo_xl.safetensors"},
    "kpop_nct_taeyong_xl": {"url": "https://civitai.com/api/download/models/121308", "filename": "kpop_nct_taeyong_xl.safetensors"},
    "kpop_nct_jaehyun_xl": {"url": "https://civitai.com/api/download/models/121312", "filename": "kpop_nct_jaehyun_xl.safetensors"},
    "kpop_nct_mark_xl": {"url": "https://civitai.com/api/download/models/122104", "filename": "kpop_nct_mark_xl.safetensors"},
    "kpop_astro_cha_eunwoo_xl": {"url": "https://civitai.com/api/download/models/121315", "filename": "kpop_astro_cha_eunwoo_xl.safetensors"},
    "kpop_exo_kai_xl": {"url": "https://civitai.com/api/download/models/121318", "filename": "kpop_exo_kai_xl.safetensors"}
}

CONTROLNET_DIR = "/controlnet_models"
CONTROLNET_MODELS = {
    "openpose": "thibaud/controlnet-openpose-sdxl-1.0",
    "canny": "diffusers/controlnet-canny-sdxl-1.0",
    "depth": "diffusers/controlnet-depth-sdxl-1.0",
}
MODEL_DIR = "/models"

# ===================================================================
# DEFINISI IMAGE & VOLUME
# ===================================================================
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]", "torch", "diffusers", "transformers",
        "accelerate", "safetensors", "Pillow", "requests", "huggingface_hub", 
        "opencv-python-headless", "controlnet_aux"
    )
)

model_volume = modal.Volume.from_name("civitai-models", create_if_missing=True)
lora_volume = modal.Volume.from_name("civitai-loras-collection-vol", create_if_missing=True)
controlnet_volume = modal.Volume.from_name("controlnet-sdxl-collection-vol", create_if_missing=True)

# ===================================================================
# FUNGSI DOWNLOAD
# ===================================================================
@app.function(image=image, volumes={MODEL_DIR: model_volume}, timeout=3600)
def download_model():
    import requests
    from pathlib import Path
    
    model_url = "https://civitai.com/api/download/models/348913?type=Model&format=SafeTensor&size=full&fp=fp16"
    model_path = Path(MODEL_DIR) / "model.safetensors"
    
    if not model_path.exists():
        print("Downloading model...")
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192): 
                f.write(chunk)
        model_volume.commit()
        print(f"Model downloaded to {model_path}")
    else:
        print(f"Model already exists at {model_path}")

@app.function(image=image, volumes={LORA_DIR: lora_volume}, timeout=3600)
def download_loras():
    import requests
    from pathlib import Path
    
    failed_downloads = []
    successful_downloads = []
    skipped_existing = []
    
    total_loras = len(LORA_MODELS)
    current_index = 0
    
    for name, data in LORA_MODELS.items():
        current_index += 1
        lora_path = Path(LORA_DIR) / data["filename"]
        
        if lora_path.exists():
            print(f"[{current_index}/{total_loras}] ✓ LoRA sudah ada: {name}")
            skipped_existing.append(name)
            continue
        
        try:
            print(f"[{current_index}/{total_loras}] 📥 Downloading LoRA: {name}...")
            response = requests.get(data["url"], stream=True, timeout=60)
            response.raise_for_status()
            
            with open(lora_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192): 
                    f.write(chunk)
            
            lora_volume.commit()
            print(f"[{current_index}/{total_loras}] ✅ LoRA berhasil diunduh: {name}")
            successful_downloads.append(name)
            
        except requests.exceptions.RequestException as e:
            print(f"[{current_index}/{total_loras}] ❌ GAGAL download LoRA: {name}")
            print(f"    Error: {str(e)}")
            failed_downloads.append({
                "index": current_index,
                "name": name,
                "filename": data["filename"],
                "error": str(e)
            })
            continue
        except Exception as e:
            print(f"[{current_index}/{total_loras}] ❌ ERROR tidak terduga: {name}")
            print(f"    Error: {str(e)}")
            failed_downloads.append({
                "index": current_index,
                "name": name,
                "filename": data["filename"],
                "error": str(e)
            })
            continue
    
    # Laporan akhir
    print("\n" + "="*70)
    print("📊 LAPORAN DOWNLOAD LORA")
    print("="*70)
    print(f"✅ Berhasil diunduh: {len(successful_downloads)}")
    print(f"⏭️  Sudah ada (diskip): {len(skipped_existing)}")
    print(f"❌ Gagal diunduh: {len(failed_downloads)}")
    print(f"📦 Total LoRA: {total_loras}")
    
    if failed_downloads:
        print("\n" + "⚠️ " * 35)
        print("DAFTAR LORA YANG GAGAL DIUNDUH:")
        print("="*70)
        for fail in failed_downloads:
            print(f"\n#{fail['index']} - {fail['name']}")
            print(f"   File: {fail['filename']}")
            print(f"   Error: {fail['error']}")
        print("\n" + "⚠️ " * 35)
    else:
        print("\n🎉 Semua LoRA berhasil diproses!")
    
    print("="*70)
    
@app.function(image=image, volumes={CONTROLNET_DIR: controlnet_volume}, timeout=3600)
def download_controlnet_models():
    from huggingface_hub import snapshot_download
    from pathlib import Path
    
    for name, repo_id in CONTROLNET_MODELS.items():
        model_dir = Path(CONTROLNET_DIR) / name
        if not model_dir.exists():
            print(f"Downloading ControlNet model: {name} from {repo_id}...")
            snapshot_download(repo_id=repo_id, local_dir=str(model_dir), local_dir_use_symlinks=False)
            controlnet_volume.commit()
            print(f"ControlNet {name} downloaded.")
    print("All ControlNet models download check complete.")


# ===================================================================
# KELAS INFERENCE
# ===================================================================
@app.cls(
    image=image,
    gpu="L4",
    volumes={MODEL_DIR: model_volume, LORA_DIR: lora_volume, CONTROLNET_DIR: controlnet_volume},
    container_idle_timeout=200
)
class ModelInference:
    @modal.enter()
    def load_model(self):
        from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, EulerDiscreteScheduler
        import torch
        
        print("Loading SDXL model...")
        model_path = f"{MODEL_DIR}/model.safetensors"
        
        self.txt2img_pipe = StableDiffusionXLPipeline.from_single_file(
            model_path, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
        )
        self.txt2img_pipe.scheduler = EulerDiscreteScheduler.from_config(self.txt2img_pipe.scheduler.config)
        self.txt2img_pipe.to("cuda")
        
        self.img2img_pipe = StableDiffusionXLImg2ImgPipeline(
            vae=self.txt2img_pipe.vae, 
            text_encoder=self.txt2img_pipe.text_encoder,
            text_encoder_2=self.txt2img_pipe.text_encoder_2, 
            tokenizer=self.txt2img_pipe.tokenizer,
            tokenizer_2=self.txt2img_pipe.tokenizer_2, 
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
        )
        self.img2img_pipe.to("cuda")
        print("✓ SDXL Model loaded successfully! Uncensored mode active.")

    def _apply_lora(self, pipe, lora_name: str, lora_scale: float):
        """Apply LoRA weights to pipeline"""
        try:
            pipe.unload_lora_weights()
        except:
            pass
            
        if lora_name and lora_name in LORA_MODELS:
            print(f"Applying LoRA: {lora_name} with scale {lora_scale}")
            lora_info = LORA_MODELS[lora_name]
            lora_path = f"{LORA_DIR}/{lora_info['filename']}"
            pipe.load_lora_weights(lora_path, adapter_name=lora_name)
            pipe.fuse_lora(lora_scale=lora_scale, adapter_names=[lora_name])

    def _preprocess_control_image(self, image, controlnet_type: str):
        """Preprocess control image based on type"""
        from controlnet_aux import OpenposeDetector, CannyDetector
        from PIL import Image
        import numpy as np
        import cv2
        
        if controlnet_type == "openpose":
            processor = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            return processor(image)
        elif controlnet_type == "canny":
            image_np = np.array(image)
            low_threshold = 100
            high_threshold = 200
            canny_image = cv2.Canny(image_np, low_threshold, high_threshold)
            canny_image = canny_image[:, :, None]
            canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
            return Image.fromarray(canny_image)
        elif controlnet_type == "depth":
            from transformers import pipeline
            depth_estimator = pipeline('depth-estimation')
            depth = depth_estimator(image)['depth']
            return depth
        else:
            return image

    @modal.method()
    def text_to_image(self, prompt: str, lora_name: str = None, lora_scale: float = 0.8, **kwargs):
        """Generate image from text prompt"""
        import torch
        
        self._apply_lora(self.txt2img_pipe, lora_name, lora_scale)
        
        enhanced_prompt = f"{prompt}, {DEFAULT_POSITIVE_PROMPT_SUFFIX}" if kwargs.get("enhance_prompt", True) else prompt
        negative_prompt = kwargs.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT
        
        generator = None
        if kwargs.get("seed", -1) != -1:
            generator = torch.Generator(device="cuda").manual_seed(kwargs.get("seed", -1))
        
        image = self.txt2img_pipe(
            prompt=enhanced_prompt, 
            negative_prompt=negative_prompt,
            num_inference_steps=kwargs.get("num_steps", 25),
            guidance_scale=kwargs.get("guidance_scale", 7.5),
            width=kwargs.get("width", 1024), 
            height=kwargs.get("height", 1024),
            generator=generator
        ).images[0]
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "image": img_str, 
            "prompt": enhanced_prompt, 
            "original_prompt": prompt,
            "negative_prompt": negative_prompt, 
            "seed": kwargs.get("seed", -1), 
            "uncensored": True
        }

    @modal.method()
    def image_to_image(
        self,
        init_image_b64: str,
        prompt: str,
        negative_prompt: str = "",
        num_steps: int = 25,
        guidance_scale: float = 7.5,
        strength: float = 0.75,
        seed: int = -1,
        enhance_prompt: bool = True,
        lora_name: str = None, 
        lora_scale: float = 0.8,
        width: int = 1024,
        height: int = 1024
    ):
        """Edit image with prompt"""
        from PIL import Image
        import torch
        
        self._apply_lora(self.img2img_pipe, lora_name, lora_scale)
        
        enhanced_prompt = f"{prompt}, {DEFAULT_POSITIVE_PROMPT_SUFFIX}" if enhance_prompt else prompt
        
        if not negative_prompt or not str(negative_prompt).strip():
            negative_prompt = DEFAULT_NEGATIVE_PROMPT
        
        print(f"Image-to-Image: {enhanced_prompt[:100]}...")
        
        init_image_bytes = base64.b64decode(init_image_b64)
        init_image = Image.open(io.BytesIO(init_image_bytes)).convert("RGB")
        init_image = init_image.resize((width, height), Image.LANCZOS)
        
        generator = None
        if seed != -1:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        
        image = self.img2img_pipe(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            strength=strength,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            width=width,
            height=height
        ).images[0]
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "image": img_str,
            "prompt": enhanced_prompt,
            "original_prompt": prompt,
            "negative_prompt": negative_prompt,
            "strength": strength,
            "seed": seed if seed != -1 else "random",
            "uncensored": True,
            "output_width": width,
            "output_height": height
        }

    @modal.method()
    def generate_with_controlnet(
        self, 
        prompt: str, 
        control_image_b64: str, 
        controlnet_type: str, 
        lora_name: str = None, 
        lora_scale: float = 0.8, 
        **kwargs
    ):
        """Generate image with ControlNet guidance"""
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
        from PIL import Image
        import torch
        
        if controlnet_type not in CONTROLNET_MODELS:
            raise ValueError(f"Invalid controlnet_type. Supported: {list(CONTROLNET_MODELS.keys())}")

        controlnet = ControlNetModel.from_pretrained(
            str(Path(CONTROLNET_DIR) / controlnet_type), 
            torch_dtype=torch.float16
        )
        
        control_pipe = StableDiffusionXLControlNetPipeline(
            vae=self.txt2img_pipe.vae, 
            text_encoder=self.txt2img_pipe.text_encoder,
            text_encoder_2=self.txt2img_pipe.text_encoder_2, 
            tokenizer=self.txt2img_pipe.tokenizer,
            tokenizer_2=self.txt2img_pipe.tokenizer_2, 
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler, 
            controlnet=controlnet,
        )
        control_pipe.to("cuda")

        self._apply_lora(control_pipe, lora_name, lora_scale)

        control_image_bytes = base64.b64decode(control_image_b64)
        control_image = Image.open(io.BytesIO(control_image_bytes)).convert("RGB")
        
        # Preprocess control image based on type
        control_image = self._preprocess_control_image(control_image, controlnet_type)
        
        enhanced_prompt = f"{prompt}, {DEFAULT_POSITIVE_PROMPT_SUFFIX}"
        negative_prompt = kwargs.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT
        
        generator = None
        if kwargs.get("seed", -1) != -1:
            generator = torch.Generator(device="cuda").manual_seed(kwargs.get("seed", -1))

        image = control_pipe(
            enhanced_prompt, 
            negative_prompt=negative_prompt, 
            image=control_image,
            num_inference_steps=kwargs.get("num_steps", 30),
            guidance_scale=kwargs.get("guidance_scale", 7.5),
            controlnet_conditioning_scale=float(kwargs.get("controlnet_scale", 0.8)),
            generator=generator,
        ).images[0]
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "image": img_str, 
            "prompt": enhanced_prompt, 
            "original_prompt": prompt,
            "negative_prompt": negative_prompt, 
            "seed": kwargs.get("seed", -1), 
            "controlnet_type": controlnet_type,
            "uncensored": True
        }


# ===================================================================
# ENDPOINT API
# ===================================================================
@app.function(image=image, secrets=[modal.Secret.from_name("custom-secret")])
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    import os
    
    web_app = FastAPI()
    
    @web_app.get("/")
    async def root():
        return {
            "service": "CivitAI Model API - Uncensored (SDXL)",
            "version": "3.2",
            "endpoints": {
                "health": "GET /health",
                "text-to-image": "POST /text2img",
                "image-to-image": "POST /img2img",
                "controlnet": "POST /controlnet"
            },
            "features": [
                "✓ No NSFW filter", 
                "✓ Uncensored generation",
                "✓ Text-to-Image (SDXL)", 
                "✓ Image-to-Image (SDXL)",
                "✓ ControlNet (SDXL)",
                "✓ Multi-LoRA support",
                "✓ GPU L4",
                "✓ Auto quality enhancement",
                "✓ Default negative prompts for best results"
            ],
            "default_prompts": {
                "positive_suffix": DEFAULT_POSITIVE_PROMPT_SUFFIX,
                "negative": DEFAULT_NEGATIVE_PROMPT
            },
            "available_loras": list(LORA_MODELS.keys()),
            "available_controlnets": list(CONTROLNET_MODELS.keys())
        }

    @web_app.get("/health")
    async def health_check():
        return {
            "status": "healthy", 
            "service": "civitai-api-fastapi",
            "mode": "uncensored-sdxl",
            "gpu": "L4"
        }

    @web_app.post("/text2img")
    async def text_to_image_endpoint(request: Request):
        try:
            data = await request.json()
            api_key = data.get("api_key")
            if api_key != os.environ.get("API_KEY"):
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            prompt = data.get("prompt")
            if not prompt: 
                raise HTTPException(status_code=400, detail="Prompt is required")
            
            lora_name = data.get("lora_name")
            lora_scale = data.get("lora_scale", 0.8)
            
            model = ModelInference()
            result = model.text_to_image.remote(
                prompt=prompt, 
                lora_name=lora_name, 
                lora_scale=lora_scale, 
                **data
            )
            return JSONResponse(content=result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @web_app.post("/img2img")
    async def image_to_image_endpoint(request: Request):
        try:
            data = await request.json()
            api_key = data.get("api_key")
            if api_key != os.environ.get("API_KEY"):
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            init_image = data.get("init_image")
            prompt = data.get("prompt")
            if not init_image or not prompt:
                raise HTTPException(status_code=400, detail="init_image and prompt are required")
            
            lora_name = data.get("lora_name")
            lora_scale = data.get("lora_scale", 0.8)
            
            model = ModelInference()
            result = model.image_to_image.remote(
                init_image_b64=init_image, 
                prompt=prompt, 
                lora_name=lora_name, 
                lora_scale=lora_scale,
                negative_prompt=data.get("negative_prompt", ""),
                num_steps=data.get("num_steps", 25),
                guidance_scale=data.get("guidance_scale", 7.5),
                strength=data.get("strength", 0.75),
                seed=data.get("seed", -1),
                enhance_prompt=data.get("enhance_prompt", True),
                width=data.get("width", 1024),
                height=data.get("height", 1024)
            )
            return JSONResponse(content=result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @web_app.post("/controlnet")
    async def controlnet_endpoint(request: Request):
        try:
            data = await request.json()
            api_key = data.get("api_key")
            if api_key != os.environ.get("API_KEY"):
                raise HTTPException(status_code=401, detail="Invalid API key")

            prompt = data.get("prompt")
            control_image = data.get("control_image")
            controlnet_type = data.get("controlnet_type")
            
            if not all([prompt, control_image, controlnet_type]):
                raise HTTPException(400, "prompt, control_image (base64), and controlnet_type are required.")
            
            lora_name = data.get("lora_name")
            lora_scale = data.get("lora_scale", 0.8)

            model = ModelInference()
            result = model.generate_with_controlnet.remote(
                prompt=prompt, 
                control_image_b64=control_image, 
                controlnet_type=controlnet_type,
                lora_name=lora_name, 
                lora_scale=lora_scale, 
                **data
            )
            return JSONResponse(content=result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return web_app

@app.local_entrypoint()
def main():
    """Local test entrypoint"""
    print("Use: modal run app.py::download_model")
    print("Use: modal run app.py::download_loras")
    print("Use: modal run app.py::download_controlnet_models")
    print("Use: modal deploy app.py")
