package edu.pengli.nlp.conference.acl2015.types;

import java.util.HashMap;

public enum Category {

	Accidents_and_Natural_Disasters(1, generateAspectsFor1()), 
	Attacks(2,generateAspectsFor2()), 
	Health_and_Safety(3, generateAspectsFor3()), 
	Endangered_Resources(4, generateAspectsFor4()), 
	Investigations_and_Trials(5,generateAspectsFor5());

	private int idx;
	private static HashMap<String, String[]> type1aspects;
	private static HashMap<String, String[]> type2aspects;
	private static HashMap<String, String[]> type3aspects;
	private static HashMap<String, String[]> type4aspects;
	private static HashMap<String, String[]> type5aspects;

	public int getId() {
		return idx;
	}

	public static HashMap<String, String[]> getAspects(int id) {
		switch (id) {
		case 1:
			return generateAspectsFor1();
		case 2:
			return generateAspectsFor2();

		case 3:
			return generateAspectsFor3();
		case 4:
			return generateAspectsFor4();
		case 5:
			return generateAspectsFor5();
		default: return null;
		}
	}

	private static HashMap<String, String[]> generateAspectsFor1() {
		type1aspects = new HashMap<String, String[]>();
		String[] keywordsForWhat = { "d", "d" };
		type1aspects.put("what", keywordsForWhat);

		String[] keywordsForWhen = { "d", "d" };
		type1aspects.put("when", keywordsForWhen);

		String[] keywordsForWhere = { "d", "d" };
		type1aspects.put("where", keywordsForWhere);

		String[] keywordsForWhy = { "d", "d" };
		type1aspects.put("why", keywordsForWhy);

		String[] keywordsForWhoaffected = { "d", "d" };
		type1aspects.put("whoaffected", keywordsForWhoaffected);

		String[] keywordsForDamages = { "d", "d" };
		type1aspects.put("damages", keywordsForDamages);

		String[] keywordsForCountermeasures = { "d", "d" };
		type1aspects.put("countermeasures", keywordsForCountermeasures);
		return type1aspects;
	}

	private static HashMap<String, String[]> generateAspectsFor2() {
		type2aspects = new HashMap<String, String[]>();
		String[] keywordsForWhat = { "riot", "violence", "raid", "ambush",
				"assault", "revolt", "violence", "massacre", "slaughter" };
		type2aspects.put("what", keywordsForWhat);

		String[] keywordsForWhen = { "january", "thursday", "today" };
		type2aspects.put("when", keywordsForWhen);

		String[] keywordsForWhere = { "location", "locality", "point",
				"position", "site" };
		type2aspects.put("where", keywordsForWhere);

		String[] keywordsForWhy = { "motive", "reason", "commit", "intent",
				"cause" };
		type2aspects.put("why", keywordsForWhy);

		String[] keywordsForWhoaffected = { "casualt", "death", "injur",
				"dead", "loss" };
		type2aspects.put("whoaffected", keywordsForWhoaffected);

		String[] keywordsForDamages = { "property", "possession", "belongings",
				"assets", "money", "economic", "loss" };
		type2aspects.put("damages", keywordsForDamages);

		String[] keywordsForCountermeasures = { "safety", "prevent",
				"hazardous", "unsafe", "maintenance", "repair", "correct" };
		type2aspects.put("countermeasures", keywordsForCountermeasures);

		String[] keywordsForPerpetrators = { "perpetrat", "commit", "culprit",
				"criminal", "incendiar" };
		type2aspects.put("perpetrators", keywordsForPerpetrators);
		return type2aspects;
	}

	private static HashMap<String, String[]> generateAspectsFor3() {
		type3aspects = new HashMap<String, String[]>();
		String[] keywordsForWhat = { "calamity", "casualty", "collision",
				"crash", "wreck", "incident", "pile", "smash", "catastrophe",
				"explosion", "nuclear", "earthquake", "eruption", "flash",
				"flood", "hurricane", "landslide", "mudslide", "slide",
				"tornado", "tsunami", "volcano", "wildfire", "blizzard" };
		type3aspects.put("what", keywordsForWhat);

		String[] keywordsForWhy = { "d", "d" };
		type3aspects.put("why", keywordsForWhy);

		String[] keywordsForWhoaffected = { "d", "d" };
		type3aspects.put("whoaffected", keywordsForWhoaffected);

		String[] keywordsForHow = { "d", "d" };
		type3aspects.put("how", keywordsForHow);

		String[] keywordsForCountermeasures = { "d", "d" };
		type3aspects.put("countermeasures", keywordsForCountermeasures);
		return type3aspects;
	}

	private static HashMap<String, String[]> generateAspectsFor4() {
		type4aspects = new HashMap<String, String[]>();
		String[] keywordsForWhat = { "d", "d" };
		type4aspects.put("what", keywordsForWhat);

		String[] keywordsForImportance = { "d", "d" };
		type4aspects.put("importance", keywordsForImportance);

		String[] keywordsForThreats = { "d", "d" };
		type4aspects.put("threats", keywordsForThreats);

		String[] keywordsForCountermeasures = { "d", "d" };
		type4aspects.put("countermeasures", keywordsForCountermeasures);
		return type4aspects;
	}

	private static HashMap<String, String[]> generateAspectsFor5() {
		type5aspects = new HashMap<String, String[]>();
		String[] keywordsForWho = { "d", "d" };
		type5aspects.put("who", keywordsForWho);

		String[] keywordsForWho_Inv = { "d", "d" };
		type5aspects.put("who_inv", keywordsForWho_Inv);

		String[] keywordsForWhy = { "d", "d" };
		type5aspects.put("why", keywordsForWhy);

		String[] keywordsForCharges = { "d", "d" };
		type5aspects.put("charges", keywordsForCharges);

		String[] keywordsForPlead = { "d", "d" };
		type5aspects.put("plead", keywordsForPlead);

		String[] keywordsForSentence = { "d", "d" };
		type5aspects.put("sentence", keywordsForSentence);
		return type5aspects;
	}

	private Category(int idx, HashMap<String, String[]> aspects) {
		this.idx = idx;
	}

	private HashMap<String, String[]> getType1Aspects() {
		return type1aspects;
	}

	private HashMap<String, String[]> getType2Aspects() {
		return type2aspects;
	}

	private HashMap<String, String[]> getType3Aspects() {
		return type3aspects;
	}

	private HashMap<String, String[]> getType4Aspects() {
		return type4aspects;
	}

	private HashMap<String, String[]> getType5Aspects() {
		return type5aspects;
	}
}
