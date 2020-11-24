#Imports
import os; import sys;
import tqdm
import requests
import mimetypes
import zipfile
import math
import pandas as pd
import ast
import matplotlib.pyplot as plt
from mplsoccer.pitch import Pitch, add_image
from mplsoccer.statsbomb import read_event, EVENT_SLUG
import numpy as np
import json
from pandas import json_normalize
import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from matplotlib import rcParams
pd.options.mode.chained_assignment = None
from scipy.ndimage import gaussian_filter
from scipy.stats import circmean
from itertools import chain
from ast import literal_eval
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import cdist 
mpl.rcParams['font.family'] = 'Slabo 27px'
anotherfont = 'Oxygen'
mpl.rcParams['figure.facecolor'] = '#082630'
mpl.rcParams['axes.facecolor'] = '#082630'
mpl.rcParams['axes.labelcolor'] = 'edece9'
mpl.rcParams['xtick.color'] = 'edece9'
mpl.rcParams['ytick.color'] = 'edece9'
mpl.rcParams['text.color'] = 'edece9' 
from highlight_text.htext import htext, fig_htext  
scattercolor='grey'
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
bu_alt = mpimg.imread('BU _ White logo.png')
import streamlit as st
# import asyncio
# loop = asyncio.new_event_loop()
# asyncio.set_event_loop(loop)

def filereader(filename):
	with open(filename, encoding="unicode_escape") as f:
#         matchdict = json.loads(f.read().encode('raw_unicode_escape').decode())
		matchdict = json.loads(f.read())
	match = json_normalize(matchdict['events'], sep="_")
	hometeam = matchdict['home']['name']
	awayteam = matchdict['away']['name']
	homeid = matchdict['home']['teamId']
	awayid = matchdict['away']['teamId']
	players = pd.DataFrame()
	homepl = json_normalize(matchdict['home']['players'],sep='_')[['name','position','shirtNo','playerId']]
	awaypl = json_normalize(matchdict['away']['players'],sep='_')[['name','position','shirtNo','playerId']]
	players = players.append(homepl)
	players = players.append(awaypl)
	# Adding playernames and pass receiver names
	match = match.merge(players,how='left')
	homedf = match[match.teamId==homeid].reset_index(drop=True)
	awaydf = match[match.teamId==awayid].reset_index(drop=True)
	homedf['receiver'] = np.where((homedf.type_displayName=='Pass')
										  &(homedf.outcomeType_displayName=='Successful'),
						  homedf.name.shift(-1),'').tolist()
	awaydf['receiver'] = np.where((awaydf.type_displayName=='Pass')
										  &(awaydf.outcomeType_displayName=='Successful'),
						  awaydf.name.shift(-1),'').tolist()
	match['receiver'] = ['' for _ in range(len(match))]
	match.loc[match.teamId==homeid,'receiver'] = homedf['receiver'].tolist()
	match.loc[match.teamId==awayid,'receiver'] = awaydf['receiver'].tolist()
	match['gameid'] = [matchdict['matchId'] for i in range(len(match))]
	return match,matchdict

def convert_to_actions(Df):
	actions = Df.copy()
	actions['second'] = actions['second'].fillna(value=0)
	actions['time_seconds'] = actions['expandedMinute']*60+actions['second']
	actions['quals'] = [{q["type"]["value"]: q["type"]["displayName"] for q in actions.qualifiers[i]} 
						for i in range(len(actions))]
	
	fieldlength=100
	fieldwidth=100
	min_dribble_length = 3
	max_dribble_length = 60
	max_dribble_duration = 10
	def add_dribbles(actions):
		next_actions = actions.shift(-1)

		same_team = actions.teamId == next_actions.teamId
		# not_clearance = actions.type_id != actiontypes.index("clearance")

		dx = actions.endX - next_actions.x
		dy = actions.endY - next_actions.y
		far_enough = dx ** 2 + dy ** 2 >= min_dribble_length ** 2
		not_too_far = dx ** 2 + dy ** 2 <= max_dribble_length ** 2

		dt = next_actions.time_seconds - actions.time_seconds
		same_phase = dt < max_dribble_duration
		same_period = actions.period_value == next_actions.period_value

		dribble_idx = same_team & far_enough & not_too_far & same_phase & same_period

		dribbles = pd.DataFrame()
		prev = actions[dribble_idx]
		nex = next_actions[dribble_idx]
		dribbles["gameid"] = nex.gameid
		dribbles["period_value"] = nex.period_value
		dribbles["id"] = prev.id + 0.1
		dribbles["time_seconds"] = (prev.time_seconds + nex.time_seconds) / 2
	#     dribbles["timestamp"] = nex.timestamp
		dribbles["teamId"] = nex.teamId
		dribbles["playerId"] = nex.playerId
		dribbles["name"] = nex.name
		dribbles["receiver"] = ''
		dribbles["x"] = prev.endX
		dribbles["y"] = prev.endY
		dribbles["endX"] = nex.x
		dribbles["endY"] = nex.y
		dribbles["quals"] = [{} for _ in range(len(dribbles))]
	#     dribbles["quals"] = {}
	#     dribbles["bodypart_id"] = bodyparts.index("foot")
		dribbles["type_displayName"] = ['Carry' for _ in range(len(dribbles))]
		dribbles["outcomeType_displayName"] = ['Successful' for _ in range(len(dribbles))]

		actions = pd.concat([actions, dribbles], ignore_index=True, sort=False)
		actions = actions.sort_values(["gameid", "period_value", "id"]).reset_index(
			drop=True
		)
		actions["id"] = range(len(actions))
		return actions

	def get_bodypart(qualifiers):
		if 15 in qualifiers:
			b = "head"
		elif 21 in qualifiers:
			b = "other"
		elif 72 in qualifiers:
			b = "leftfoot"
		elif 20 in qualifiers:
			b = "rightfoot"
		else:
			b = ''
		return b

	def get_result(args):
		e, outcome, q = args
		if e == "OffsidePass":
			r = "Unsuccessful"  # offside
		elif e in ["SavedShot", "MissedShots", "ShotOnPost"]:
			r = "Unsuccessful"
		elif e == "Goal":
			if 28 in q:
				r = "OwnGoal"  # own goal, x and y must be switched
			else:
				r = "Successful"
		elif outcome == 'Successful':
			r = "Successful"
		else:
			r = "Unsuccessful"
		return r

	def get_type(args):
		eventname, outcome, q = args
		if eventname == "Pass" or eventname == "OffsidePass":
			cross = 2 in q
			freekick = 5 in q
			corner = 6 in q
			throw_in = 107 in q
			if throw_in:
				a = "throw_in"
			elif freekick and cross:
				a = "freekick_crossed"
			elif freekick:
				a = "freekick_short"
			elif corner and cross:
				a = "corner_crossed"
			elif corner:
				a = "corner_short"
			elif cross:
				a = "cross"
			else:
				a = "Pass"
		elif eventname == "TakeOn":
			a = "TakeOn"
		elif eventname == "Foul" and outcome == 'Unsuccessful':
			a = "Foul"
		elif eventname == "Tackle":
			a = "Tackle"
		elif eventname == "Interception" or eventname == "BlockedPass":
			a = "Interception"
		elif eventname in ["MissedShots", "ShotOnPost", "SavedShot", "Goal"]:
			if 9 in q:
				a = "Penalty"
			elif 26 in q:
				a = "Freekick"
			else:
				a = "Shot"
		elif eventname == "Save":
			a = "Save"
		elif eventname == "Claim":
			a = "Claim"
		elif eventname == "Punch":
			a = "Punch"
		elif eventname == "KeeperPickup":
			a = "KeeperPickup"
		elif eventname == "Clearance":
			a = "Clearance"
		elif eventname == "BallTouch" and outcome == 'Unsuccessful':
			a = "BadTouch"
		elif eventname == 'Carry':
			a = "Carry"
		elif eventname == 'BallRecovery':
			a = "BallRecovery"
		else:
			a = "NonAction"
		return a

	def fix_owngoal_coordinates(actions):
		owngoals_idx = ((actions.outcome == "OwnGoal")&(actions.events == "Shot"))
		actions.loc[owngoals_idx, "x"] = fieldlength - actions[owngoals_idx].x.values
		actions.loc[owngoals_idx, "y"] = fieldwidth - actions[owngoals_idx].y.values
		actions.loc[owngoals_idx, "goalMouthY"] = fieldwidth - actions[owngoals_idx].goalMouthY.values
		actions.loc[owngoals_idx, "endY"] = fieldwidth - actions[owngoals_idx].goalMouthY.values
		actions.loc[owngoals_idx, "endX"] = fieldlength
		return actions

	def fix_clearances(actions):
		next_actions = actions.shift(-1)
		next_actions[-1:] = actions[-1:]
		clearance_idx = actions.events == "Clearance"
		actions.loc[clearance_idx, "endX"] = next_actions[clearance_idx].x.values
		actions.loc[clearance_idx, "endY"] = next_actions[clearance_idx].y.values
		return actions

	def savedindices(actions):
		next_actions = actions.shift(-1)
		next_actions[-1:] = actions[-1:]
		saved_idx = actions.type_displayName == "SavedShot"
		actions.loc[saved_idx, "endX"] = fieldlength - next_actions[saved_idx].x.values
		actions.loc[saved_idx, "endY"] = fieldwidth - next_actions[saved_idx].y.values
		return actions

	def throughball(qualifiers):
		if 4 in qualifiers:
			b = 1
		else:
			b = 0
		return b
	def kp(qualifiers):
		if 11113 in qualifiers:
			b = 1
		else:
			b = 0
		return b
	def assist(qualifiers):
		if 11111 in qualifiers:
			b = 1
		else:
			b = 0
		return b
	def goalkick(qualifiers):
		if 124 in qualifiers:
			b = 1
		else:
			b = 0
		return b
	def gkthrow(qualifiers):
		if 123 in qualifiers:
			b = 1
		else:
			b = 0
		return b

	actions = add_dribbles(actions)
	actions['bodypart'] = actions['quals'].apply(get_bodypart)
	actions["outcome"] = actions[["type_displayName", "outcomeType_displayName", "quals"]].apply(get_result, axis=1)
	actions["events"] = actions[["type_displayName", "outcomeType_displayName", "quals"]].apply(get_type, axis=1)
	actions = fix_owngoal_coordinates(actions)
	actions = fix_clearances(actions)
	actions = savedindices(actions)
	actions['TB'] = actions['quals'].apply(throughball)
	actions['KP'] = actions['quals'].apply(kp)
	actions['Assist'] = actions['quals'].apply(assist)
	actions['GK'] = actions['quals'].apply(goalkick)
	actions['GKthrow'] = actions['quals'].apply(gkthrow)
	for col in ["x", "endX", "blockedX"]:
		actions[col] = actions[col] / fieldlength * 120
	for col in ["y", "endY", "goalMouthY", "blockedY"]:
		actions[col] = actions[col] / fieldwidth * 80
	actions = actions.sort_values(["gameid", "period_value", "time_seconds"])
	actions.reset_index(drop=True, inplace=True)
	
	return actions[['gameid','period_value','time_seconds','expandedMinute','x','y','endX','endY','blockedX','blockedY','goalMouthY',
					'quals','type_displayName','outcomeType_displayName','name','receiver','position','shirtNo','playerId',
					'teamId','KP','Assist','TB','GK','GKthrow','isTouch','bodypart','outcome','events']]

def passtypes(Df,team,teamid):
    df = Df.copy()
    # matchdict = Dict
    pitch = Pitch(pitch_type='statsbomb', figsize=(11,6), line_zorder=2, layout=(2,3), view='half',
                  line_color='#c7d5cc', orientation='vertical',constrained_layout=True, tight_layout=False)
    fig, ax = pitch.draw()    
    
    passdf = df.query("(type_displayName == 'Pass')&(outcomeType_displayName=='Successful')\
                         &(teamId==@teamid)").reset_index()
    
    passdf['y'] = 80 - passdf['y']
    passdf['endY'] = 80 - passdf['endY']

    progtocentre = passdf.query("(x<80)&(endX>80)&(endY>18)&(endY<62)")[['x','y','endX','endY']]
    wingplay = passdf.query("(x<80)&(endX>80)&((endY<18)or(endY>62))")[['x','y','endX','endY']]
    allbox = passdf.query("(endX>102)&(endY>18)&(endY<62)")[['x','y','endX','endY']]
    wingtobox = passdf.query("(x>80)&((y<18)or(y>62))&(endX>102)&(endY>18)&(endY<62)")[['x','y','endX','endY']]
    halfspaceplay = passdf.query("(x>80)&(((y>18)&(y<30))or((y>50)&(y<62)))&(endX>x)")[['x','y','endX','endY']]
    zone14play = passdf.query("(x>80)&(x<102)&(y>30)&(y<50)&(endX>x)")[['x','y','endX','endY']]

    pitch.lines(progtocentre.x, progtocentre.y, progtocentre.endX, progtocentre.endY, transparent=True, lw=2, 
                comet=True, color='#a1d76a',ax=ax[0,0])
    ax[0,0].set_title('Middle 3rd'+'\n'+'to Centre of final 3rd',fontsize=15)
    pitch.lines(wingplay.x, wingplay.y, wingplay.endX, wingplay.endY, transparent=True, lw=2, 
                comet=True, color='#a1d76a',ax=ax[0,1])
    ax[0,1].set_title('Middle 3rd'+'\n'+'to wing of final 3rd',fontsize=15)
    pitch.lines(allbox.x, allbox.y, allbox.endX, allbox.endY, transparent=True, lw=2, comet=True,
                color='#a1d76a',ax=ax[0,2])
    ax[0,2].set_title('All passes to box',fontsize=15)
    pitch.lines(wingtobox.x, wingtobox.y, wingtobox.endX, wingtobox.endY, transparent=True, lw=2, comet=True,
                color='#a1d76a',ax=ax[1,0])
    ax[1,0].set_title('Wing to box',fontsize=15)
    pitch.lines(halfspaceplay.x, halfspaceplay.y, halfspaceplay.endX, halfspaceplay.endY, transparent=True, 
                lw=2, comet=True, color='#a1d76a',ax=ax[1,1])
    ax[1,1].set_title('Attacking Passes'+'\n'+'From halfspace',fontsize=15)
    pitch.lines(zone14play.x, zone14play.y, zone14play.endX, zone14play.endY, transparent=True, lw=2, comet=True, 
                color='#a1d76a',ax=ax[1,2])
    ax[1,2].set_title('Attacking Passes'+'\n'+'From Zone 14',fontsize=15)
    fig.suptitle(team+' Progressive/Attacking Passes',fontsize=20)
    fig.text(0.05, -0.05, "Created by Soumyajit Bose / @Soumyaj15209314",fontstyle="italic",fontsize=15,color='#edece9')
    add_image(bu_alt, fig, left=0.9, bottom=0.9, width=0.07)
    return fig

def PPDAcalculator(Df,homeid,awayid):
    home = Df[(Df.teamId==homeid)]
    away = Df[(Df.teamId==awayid)]
    homedef = len(home.query("(type_displayName in ['Tackle','Interception','Challenge'])&(x>48)"))
    homepass = len(home.query("(type_displayName=='Pass')&(x<72)&(outcomeType_displayName=='Successful')"))
    homefouls = len(home.query("(type_displayName=='Foul')&(outcomeType_displayName=='Unsuccessful')&(x>48)"))
    awaydef = len(away.query("(type_displayName in ['Tackle','Interception','Challenge'])&(x>48)"))
    awaypass = len(away.query("(type_displayName=='Pass')&(x<72)&(outcomeType_displayName=='Successful')"))
    awayfouls = len(away.query("(type_displayName=='Foul')&(outcomeType_displayName=='Unsuccessful')&(x>48)"))
    return round(awaypass/(homedef+homefouls)),round(homepass/(awaydef+awayfouls))

def datatable(Df,hometeam,homeid,awayteam,awayid):
    hg = len(Df[(Df.teamId==homeid)&(Df.type_displayName=='Goal')])
    hsh = len(Df.query("(teamId==@homeid)&(type_displayName in ['SavedShot', 'ShotOnPost', 'MissedShots', 'Goal'])"))
    hsot = len(Df.query("(teamId==@homeid)&(type_displayName in ['SavedShot'])&(goalMouthY<=44)&(goalMouthY>=36)&\
                (blockedX>=116)"))+hg
    hp = len(Df[(Df.teamId==homeid)&(Df.type_displayName=='Pass')])
    hps = len(Df[(Df.teamId==homeid)&(Df.type_displayName=='Pass')
                      &(Df.outcomeType_displayName=='Successful')])
    hpp = round(hps*100/hp)
    hppda = PPDAcalculator(Df,homeid,awayid)[0]
    ag = len(Df[(Df.teamId==awayid)&(Df.type_displayName=='Goal')])
    ash = len(Df.query("(teamId==@awayid)&(type_displayName in ['SavedShot', 'ShotOnPost', 'MissedShots', 'Goal'])"))
    asot = len(Df.query("(teamId==@awayid)&(type_displayName in ['SavedShot'])&(goalMouthY<=44)&(goalMouthY>=36)&\
                (blockedX>=116)"))+ag
    ap = len(Df[(Df.teamId==awayid)&(Df.type_displayName=='Pass')])
    aps = len(Df[(Df.teamId==awayid)&(Df.type_displayName=='Pass')
                      &(Df.outcomeType_displayName=='Successful')])
    app = round(aps*100/ap)
    hposs = round(hp*100/(hp+ap)) 
    aposs = round(100 - hposs)
    appda = PPDAcalculator(Df,homeid,awayid)[1]

    datalist = [[hg,ag],[hsh,ash],[hsot,asot],[hp,ap],[hps,aps],[hposs,aposs],[hpp,app],[hppda,appda]]
    datanamelist = ['Goals','Total Shots','Total Shots on Target','Total Passes',
                   'Successful Passes','Possession %','Pass Completion %','PPDA']
    teamlist = [hometeam,awayteam]
    displaydf = pd.DataFrame(data=datalist, index=datanamelist, columns=teamlist).astype(int)
    title_text = 'Game data at a glance'
#     fig_border = '#edece9'
    fig = plt.figure(linewidth=2,
#        edgecolor=fig_border,
       tight_layout={'pad':1},
#        figsize=(10,6)
      )
    the_table = plt.table(cellText=displaydf.values,
                          cellColours=[['#082630','#082630'] for i in range(8)],
                          rowLabels=displaydf.index,
                          rowLoc='right',
                          colLabels=displaydf.columns,
                          rowColours=['#082630' for i in range(8)],
                          colColours=['#082630' for i in range(2)],
                          loc='center',
                         fontsize=25)
    the_table.set_fontsize(20)
    the_table.scale(1, 1.5)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.box(on=None)
    plt.suptitle(title_text,fontsize=25)
    plt.draw()
    fig.text(0.05, 0.05, "Created by Soumyajit Bose / @Soumyaj15209314",fontstyle="italic",fontsize=15,color='#edece9')
    # fig.text(0.05, -0.01, "Data from fbref and Understat",
    #          fontstyle="italic",fontsize=15,color='#edece9')
    add_image(bu_alt, fig, left=0.9, bottom=0, width=0.07)
    return fig

def defensesmap(Df,team,teamid):
    df = Df.copy()
    pitch = Pitch(pitch_type='statsbomb', figsize=(10,6), line_zorder=2, layout=(1,2),
                  line_color='k', orientation='horizontal',constrained_layout=True, tight_layout=False)
    fig, ax = pitch.draw()
    
    defensedf = df.query("(type_displayName in ['Aerial','Clearance','Interception','Tackle','BlockedPass','Challenge'])\
                         &(teamId==@teamid)").reset_index()
    defensedf['defensive'] = [1 for i in range(len(defensedf))]
    for i in range(len(defensedf)):
        if(defensedf.loc[i,['type_displayName']][0]=='Aerial'):
            quals = defensedf.quals[i]
            if(286 in quals):
                defensedf.loc[i,['defensive']] = 0
        if(defensedf.loc[i,['type_displayName']][0]=='Challenge'):
            quals = defensedf.quals[i]
            if(286 in quals):
                defensedf.loc[i,['defensive']] = 0
    defensedf = defensedf[defensedf.defensive==1]
    defensedf['y'] = 80 - defensedf['y']
    ppda = defensedf[defensedf.x>=48]
    ppda = ppda.query("type_displayName in ['Interception','Tackle','Challenge']")
    deepdf = defensedf[defensedf.x<48]
    bin_statistic = pitch.bin_statistic(ppda.x, ppda.y, statistic='count', bins=(25, 25))
    bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
    pcm = pitch.heatmap(bin_statistic, ax=ax[0], cmap='Reds', edgecolors=None)
    ax[0].set_title(team+'\n'+' Pressurizing'+'\n'+ 'Defensive activities',fontsize=30)
    bin_statistic = pitch.bin_statistic(deepdf.x, deepdf.y, statistic='count', bins=(25, 25))
    bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
    pcm = pitch.heatmap(bin_statistic, ax=ax[1], cmap='Reds', edgecolors=None)
    # cbar = fig.colorbar(pcm, ax=ax[1])
    ax[1].set_title(team+'\n'+' Goal Protecting'+'\n'+'Defensive activities',fontsize=30)
    fig.text(0.0, 0.0, "Created by Soumyajit Bose / @Soumyaj15209314",fontstyle="italic",fontsize=15,color='#edece9')
    fig.text(0.0, 0.15, "Pressurizing defensive activities include tackles, interceptions and challenges"+'\n'+
     "made high up in opposition territory",fontstyle="italic",fontsize=15,color='#edece9')
    fig.text(0.0, 0.05, "Goal protecting defensive activities include tackles, interceptions, defensive aerials, challenges,"
        +'\n'+"clearances and blocked passes deep in own territory",fontstyle="italic",fontsize=15,color='#edece9')
    add_image(bu_alt, fig, left=0.9, bottom=0.01, width=0.07)
    return fig

def badpasses(Df,hometeam,homeid):
    df = Df.copy()
    pitch = Pitch(pitch_type='statsbomb', figsize=(10,4.5), line_zorder=2, layout=(1,2),
                  line_color='k', orientation='horizontal',constrained_layout=True, tight_layout=False)
    fig, ax = pitch.draw()
    
    def3rd = df.query("(teamId==@homeid)&(type_displayName == 'Pass')&(x<40)")
    mid3rd = df.query("(teamId==@homeid)&(type_displayName == 'Pass')&(x>=40)&(x<80)")
    defmask = def3rd.outcomeType_displayName == 'Successful'
    midmask = mid3rd.outcomeType_displayName == 'Successful'
    # awaymask = awaydef.outcomeType_displayName == 'Successful'

    pitch.arrows(def3rd[~defmask].x,80-def3rd[~defmask].y,def3rd[~defmask].endX,80-def3rd[~defmask].endY,
        headwidth=5, headlength=5,zorder=5,color='tab:red',width=1,ax=ax[0])
    # pitch.lines(homedef[~homemask].x,80-homedef[~homemask].y,homedef[~homemask].endX,80-homedef[~homemask].endY,
    #             zorder=5,color='tab:red',linewidth=2,comet=True,ax=ax[0])
    
    pitch.arrows(mid3rd[~midmask].x,80-mid3rd[~midmask].y,mid3rd[~midmask].endX,80-mid3rd[~midmask].endY,
        headwidth=5, headlength=5,zorder=5,color='tab:red',width=1,ax=ax[1])
    # pitch.lines(awaydef[~awaymask].x,80-awaydef[~awaymask].y,awaydef[~awaymask].endX,80-awaydef[~awaymask].endY,
    #             zorder=5,color='tab:red',linewidth=2,comet=True,ax=ax[1])

    ax[0].set_title(hometeam+' Unsuccessful Passes'+'\n'+'from defensive 3rd',fontsize=20)        
    ax[1].set_title(hometeam+' Unsuccessful Passes'+'\n'+'from middle 3rd',fontsize=20)
    # fig.text(0.0, 0.05, "All unsuccessful passes originating from defensive third of the pitch",fontstyle="italic",fontsize=15,color='#edece9')
    fig.text(0.0, 0.0, "Created by Soumyajit Bose / @Soumyaj15209314",fontstyle="italic",fontsize=15,color='#edece9')
    add_image(bu_alt, fig, left=0.9, bottom=0.0, width=0.07)
    return fig

def turnovermap(Df,hometeam,homeid,awayteam,awayid):
    df = Df.copy()
    pitch = Pitch(pitch_type='statsbomb', figsize=(10,6), line_zorder=2, layout=(1,2),
                  line_color='k', orientation='horizontal',constrained_layout=True, tight_layout=False)
    fig, ax = pitch.draw()
    
    homedef = df.query("(teamId==@homeid)&\
                            (type_displayName in ['Clearance','Tackle','Interception','Aerial','BlockedPass'])&\
                            (outcomeType_displayName=='Successful')")
    awaydef = df.query("(teamId==@awayid)&\
                            (type_displayName in ['Clearance','Tackle','Interception','Aerial','BlockedPass'])&\
                            (outcomeType_displayName=='Successful')")

    for i in homedef.index.tolist():
    	if(i<len(df)-1):
	        if((df.teamId[i+1]==homeid)&(df.type_displayName[i+1] in ['BallRecovery','Pass','TakeOn'])&
	            (df.outcomeType_displayName[i+1]=='Successful')):
	            pitch.scatter(homedef.x[i],80-homedef.y[i],marker='o',s=50,zorder=5,facecolors='#082630',
	                          edgecolors='#fee090',linewidth=3,ax=ax[0])
    for i in awaydef.index.tolist():
    	if(i<len(df)-1):
	        if((df.teamId[i+1]==homeid)&(df.type_displayName[i+1] in ['BallRecovery','Pass','TakeOn'])&
	            (df.outcomeType_displayName[i+1]=='Successful')):
	            pitch.scatter(awaydef.x[i],80-awaydef.y[i],marker='o',s=50,zorder=5,facecolors='#082630',
	                          edgecolors='#fee090',linewidth=3,ax=ax[1])
    ax[0].set_title(hometeam+'\n'+'Turnover creating'+'\n'+'Defensive activities',fontsize=30)        
    ax[1].set_title(awayteam+'\n'+'Turnover creating'+'\n'+'Defensive activities',fontsize=30)
    fig.text(0.0, 0.0, "Created by Soumyajit Bose / @Soumyaj15209314",fontstyle="italic",fontsize=15,color='#edece9')
    fig.text(0.0, 0.05, "Turnover creating defensive activities include tackles, interceptions, aerials, clearances,"
        +'\n'+" and blocked passes leading to winning possession",fontstyle="italic",fontsize=15,color='#edece9')
    add_image(bu_alt, fig, left=0.9, bottom=0.01, width=0.07)
    return fig

def keypasses(Df,hometeam,homeid,awayteam,awayid):
    df = Df.copy()
    pitch = Pitch(pitch_type='statsbomb', figsize=(10,5), line_zorder=2, layout=(1,2),
                  line_color='k', orientation='horizontal',constrained_layout=True, tight_layout=False)
    fig, ax = pitch.draw()
    
    homedef = df.query("(teamId==@homeid)&(KP == 1)&(endX!=0)&(endY!=0)")
    awaydef = df.query("(teamId==@awayid)&(KP == 1)&(endX!=0)&(endY!=0)")
    # homemask = homedef.outcomeType_displayName == 'Successful'
    # awaymask = awaydef.outcomeType_displayName == 'Successful'

    pitch.arrows(homedef.x,80-homedef.y,homedef.endX,80-homedef.endY,headwidth=5, headlength=5,
                zorder=5,color='#fee090',width=1,ax=ax[0])
    # pitch.lines(homedef[~homemask].x,80-homedef[~homemask].y,homedef[~homemask].endX,80-homedef[~homemask].endY,
    #             zorder=5,color='tab:red',linewidth=2,comet=True,ax=ax[0])
    
    pitch.arrows(awaydef.x,80-awaydef.y,awaydef.endX,80-awaydef.endY,headwidth=5, headlength=5,
                zorder=5,color='#fee090',width=1,ax=ax[1])
    # pitch.lines(awaydef[~awaymask].x,80-awaydef[~awaymask].y,awaydef[~awaymask].endX,80-awaydef[~awaymask].endY,
    #             zorder=5,color='tab:red',linewidth=2,comet=True,ax=ax[1])

    ax[0].set_title(hometeam+'\n'+'Key Passes',fontsize=30)        
    ax[1].set_title(awayteam+'\n'+'Key Passes',fontsize=30)
    fig.text(0.0, 0.0, "Created by Soumyajit Bose / @Soumyaj15209314",fontstyle="italic",fontsize=15,color='#edece9')
    add_image(bu_alt, fig, left=0.9, bottom=0.01, width=0.07)
    return fig

def fieldtilt(Df,homeid,awayid):
    home = Df[(Df.teamId==homeid)]
    away = Df[(Df.teamId==awayid)]
    homepass = len(home.query("(type_displayName=='Pass')&(x>80)&(outcomeType_displayName=='Successful')"))
    awaypass = len(away.query("(type_displayName=='Pass')&(x>80)&(outcomeType_displayName=='Successful')"))
    return round(homepass/(homepass+awaypass)*100),round(awaypass/(awaypass+homepass)*100)

def plotter(Df,hometeam,homeid,awayteam,awayid):
	df = Df.copy()
	# ppdalist = [PPDAcalculator(df.query("(time_seconds>=900*@i)&(time_seconds<=900*(@i+1))"),homeid,awayid) 
 #                for i in range(6)]
	timelist = ['0-15','15-30','30-45','45-60','60-75','75-90']
	fieldtiltlist = [fieldtilt(df.query("(time_seconds>=900*@i)&(time_seconds<=900*(@i+1))"),homeid,awayid) 
	                for i in range(6)]
	fig,ax = plt.subplots(figsize=(7,4))
	fig.set_facecolor('#082630')
	ax.set_facecolor('#082630')
	hc = 'tab:red'
	ac = 'orange'
	ax.scatter(timelist,[i[0] for i in fieldtiltlist],marker='h',s=150,c='gold',zorder=3)
	ax.plot(timelist,[i[0] for i in fieldtiltlist],c=hc)
	ax.fill_between(timelist,[i[0] for i in fieldtiltlist],color=hc,alpha=0.3)
	n_lines = 10
	diff_linewidth = 1.05
	alpha_value = 0.03
	for n in range(1, n_lines+1):
	    ax.plot(timelist,[i[0] for i in fieldtiltlist],c=hc,
	            linewidth=2+(diff_linewidth*n),
	            alpha=alpha_value)
	ax.set_xticklabels(timelist,color='#edece9')
	ax.set_ylabel("Field tilt",fontsize=15,color='#edece9')
	ax.set_xlabel("Time intervals",fontsize=15,color='#edece9')
	ax.yaxis.label.set_color('#edece9')
	ax.tick_params(axis='y', colors='#edece9')
	# ax.set_title(hometeam+" Field tilt, "+"Mean : "+str(fieldtilt(newmatch)[0])+'\n'+
	#                 awayteam+" Field tilt, "+"Mean : "+str(fieldtilt(newmatch)[1]),fontsize=20,color='#edece9')

	ax.scatter(timelist,[i[1] for i in fieldtiltlist],marker='h',s=150,c='gold',zorder=3)
	ax.plot(timelist,[i[1] for i in fieldtiltlist],c=ac)
	ax.fill_between(timelist,[i[1] for i in fieldtiltlist],color=ac,alpha=0.3)
	for n in range(1, n_lines+1):
	    ax.plot(timelist,[i[1] for i in fieldtiltlist],c=ac,
	            linewidth=2+(diff_linewidth*n),
	            alpha=alpha_value)

	ax.set_xticklabels(timelist,color='#edece9')
	ax.set_xlabel("Time intervals",fontsize=15,color='#edece9')
	ax.set_ylabel("Field tilt",fontsize=15,color='#edece9')
	ax.yaxis.label.set_color('#edece9')
	ax.set_ylim(0,100)
	# ax.set_xlim(0,100)
	ax.tick_params(axis='y', colors='#edece9')
	# ax.set_title(awayteam+" Field tilt "+'\n'+"Mean : "+str(fieldtilt(newmatch)[1]),fontsize=20,color='#edece9')
	fig_htext(s = f"<{hometeam}> Field tilt, Mean : "+str(fieldtilt(df,homeid,awayid)[0])+'\n'+
	              f"<{awayteam}> Field tilt, Mean : "+str(fieldtilt(df,homeid,awayid)[1]),
	       x = 0.25, y = 0.97, highlight_colors = [hc,ac],
	         highlight_weights=['bold','bold'],string_weight='bold',fontsize=20,color='#edece9')
	            
	fig.text(0.05, -0.025, "Created by Soumyajit Bose / @Soumyaj15209314",fontstyle="italic",fontsize=9,color='#edece9')
	add_image(bu_alt, fig, left=0.95, bottom=0.97, width=0.05)
	plt.tight_layout()
	return fig

def playerpasses(Df,player):
    df = Df.copy()
    df = df[df.name==player]
    df = df[(df.type_displayName=='Pass')&(df.outcomeType_displayName=='Successful')]
#     df = tags(df)
    df['y'] = 80 - df['y']
    df['endY'] = 80 - df['endY']
    df['dist1'] = np.sqrt((120-df.x)**2 + (40-df.y)**2)
    df['dist2'] = np.sqrt((120-df.endX)**2 + (40-df.endY)**2)
    df['distdiff'] = df['dist1'] - df['dist2']
#     df = df.query("(throwintag==0)&(freekicktag==0)&(cornertag==0)")
    pass1 = df.query("(x<60)&(endX<60)&(distdiff>=30)")
    pass2 = df.query("(x<60)&(endX>60)&(distdiff>=15)")
    pass3 = df.query("(x>60)&(endX>60)&(distdiff>=10)")
    pass1 = pass1.append(pass2)
    pass1 = pass1.append(pass3)
    return pass1

def progpassplotter(Df,team,teamid):
	pitch = Pitch(pitch_type='statsbomb', figsize=(12,10), line_zorder=2, layout=(4,4), view='full',
                  line_color='#c7d5cc', orientation='horizontal',constrained_layout=True, tight_layout=False)
	fig, ax = pitch.draw()
	df = Df[(Df.teamId==teamid)&(Df.name!='')].reset_index()
	players = df.name.unique().tolist()
	Nplayers = len(players)
	for i in range(Nplayers):
		p = players[i]
		Passp = playerpasses(df,p)
		pitch.lines(Passp.x, Passp.y, Passp.endX, Passp.endY, transparent=True, lw=2, 
			comet=True, color='#a1d76a',ax=ax[i//4,i%4])
		ax[i//4,i%4].set_title(p,fontsize=15)

	for i in range(Nplayers,16):
		ax[i//4,i%4].remove()

	fig.text(0.05, -0.04, "Created by Soumyajit Bose / @Soumyaj15209314",fontstyle="italic",fontsize=15,color='#edece9')
	add_image(bu_alt, fig, left=0.85, bottom=-0.07, width=0.05)
	fig_htext(s = "<Completed Progressive Passes> by all players of "+team,
	          x = 0.1, y = 1, highlight_colors = ['#a1d76a'],
	          highlight_weights=['bold'],
	          string_weight='bold',fontsize=20,color='#edece9')
	return fig

def switches(Df,hometeam,homeid,awayteam,awayid):
    df = Df.copy()
    pitch = Pitch(pitch_type='statsbomb', figsize=(10,5), line_zorder=2, layout=(1,2),
                  line_color='k', orientation='horizontal',constrained_layout=True, tight_layout=False)
    fig, ax = pitch.draw()
    
    homedef = df.query("(teamId==@homeid)&(type_displayName == 'Pass')&((endY-y)**2>=1600.0)")
    awaydef = df.query("(teamId==@awayid)&(type_displayName == 'Pass')&((endY-y)**2>=1600.0)")
    homemask = homedef.outcomeType_displayName == 'Successful'
    awaymask = awaydef.outcomeType_displayName == 'Successful'

    pitch.lines(homedef[homemask].x,80-homedef[homemask].y,homedef[homemask].endX,80-homedef[homemask].endY,
                zorder=5,color='#fee090',linewidth=2,comet=True,ax=ax[0])
    pitch.lines(homedef[~homemask].x,80-homedef[~homemask].y,homedef[~homemask].endX,80-homedef[~homemask].endY,
                zorder=5,color='tab:red',linewidth=2,comet=True,ax=ax[0])
    
    pitch.lines(awaydef[awaymask].x,80-awaydef[awaymask].y,awaydef[awaymask].endX,80-awaydef[awaymask].endY,
                zorder=5,color='#fee090',linewidth=2,comet=True,ax=ax[1])
    pitch.lines(awaydef[~awaymask].x,80-awaydef[~awaymask].y,awaydef[~awaymask].endX,80-awaydef[~awaymask].endY,
                zorder=5,color='tab:red',linewidth=2,comet=True,ax=ax[1])

    ax[0].set_title(hometeam+'\n'+'Switches',fontsize=30)        
    ax[1].set_title(awayteam+'\n'+'Switches',fontsize=30)
    fig.text(0.0, 0.0, "Created by Soumyajit Bose / @Soumyaj15209314",fontstyle="italic",fontsize=15,color='#edece9')
    fig.text(0.0, 0.05, "Red lines for unsuccessful switches, yellow for successful ones",
                                                        fontstyle="italic",fontsize=15,color='#edece9')
    add_image(bu_alt, fig, left=0.9, bottom=0.01, width=0.07)
    return fig

def takeonmap(Df,hometeam,homeid,awayteam,awayid):
    df = Df.copy()
    pitch = Pitch(pitch_type='statsbomb', figsize=(10,5), line_zorder=2, layout=(1,2),
                  line_color='k', orientation='horizontal',constrained_layout=True, tight_layout=False)
    fig, ax = pitch.draw()
    
    homedef = df.query("(teamId==@homeid)&(type_displayName == 'TakeOn')")
    awaydef = df.query("(teamId==@awayid)&(type_displayName == 'TakeOn')")
    homemask = homedef.outcomeType_displayName == 'Successful'
    awaymask = awaydef.outcomeType_displayName == 'Successful'

    pitch.scatter(homedef[homemask].x,80-homedef[homemask].y,marker='o',s=50,zorder=5,facecolors='#082630',
                          edgecolors='#fee090',linewidth=3,ax=ax[0])
    pitch.scatter(homedef[~homemask].x,80-homedef[~homemask].y,marker='o',s=50,zorder=5,facecolors='#082630',
                          edgecolors='tab:red',linewidth=3,ax=ax[0])
    
    pitch.scatter(awaydef[awaymask].x,80-awaydef[awaymask].y,marker='o',s=50,zorder=5,facecolors='#082630',
                          edgecolors='#fee090',linewidth=3,ax=ax[1])
    pitch.scatter(awaydef[~awaymask].x,80-awaydef[~awaymask].y,marker='o',s=50,zorder=5,facecolors='#082630',
                          edgecolors='tab:red',linewidth=3,ax=ax[1])

    ax[0].set_title(hometeam+'\n'+'TakeOns',fontsize=30)        
    ax[1].set_title(awayteam+'\n'+'TakeOns',fontsize=30)
    fig.text(0.0, 0.0, "Created by Soumyajit Bose / @Soumyaj15209314",fontstyle="italic",fontsize=15,color='#edece9')
    fig.text(0.0, 0.05, "Red markers for unsuccessful takeons, yellow for successful ones",
                                                        fontstyle="italic",fontsize=15,color='#edece9')
    add_image(bu_alt, fig, left=0.9, bottom=0.01, width=0.07)
    return fig

def touchmap(Df,team,teamid):
    df = Df.copy()
    pitch = Pitch(pitch_type='statsbomb', figsize=(12,7), line_zorder=2, layout=(4,4),
                  line_color='k', orientation='horizontal',constrained_layout=True, tight_layout=False)
    fig, ax = pitch.draw()
    
    touchdf = df[(df.isTouch==True)&(df.teamId==teamid)&(df.name!='')].reset_index()
    touchdf['y'] = 80 - touchdf['y']
    players = touchdf.name.unique().tolist()
    Nplayers = len(players)
    for i in range(Nplayers):
        pdf = touchdf[touchdf.name==players[i]]
        bin_statistic = pitch.bin_statistic(pdf.x, pdf.y, statistic='count', bins=(25, 25))
        bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
        pcm = pitch.heatmap(bin_statistic, ax=ax[i//4,i%4], cmap='Reds', edgecolors=None)
        ax[i//4,i%4].set_title(players[i],fontsize=15)
    for i in range(Nplayers,16):
        ax[i//4,i%4].remove()
    
    fig.text(0.0, 0.0, "Created by Soumyajit Bose / @Soumyaj15209314",fontstyle="italic",fontsize=15,color='#edece9')
    fig.text(0.3, 1.01, team+" Touch-based heatmaps",fontsize=25,color='#edece9')
    add_image(bu_alt, fig, left=0.95, bottom=0.97, width=0.07)
    return fig

def passmap(Df,teamname,teamid,min1,max1):
	pitch = Pitch(pitch_type='statsbomb', figsize=(20,13), line_zorder=2, 
			  line_color='#c7d5cc', orientation='horizontal',constrained_layout=True, tight_layout=False)
	fig, ax = pitch.draw()
	df = Df.copy()
	df = df[(df.expandedMinute>=min1)&(df.expandedMinute<=max1)]    
	allplayers = df[(df.teamId==teamid)&(df.name!='')].name.tolist()
	playersubbedoff = df[(df.type_displayName == 'SubstitutionOff')&(df.teamId==teamid)]['name'].tolist()
	timeoff = df[(df.type_displayName == 'SubstitutionOff')&(df.teamId==teamid)]['expandedMinute'].tolist()
	playersubbedon = df[(df.type_displayName == 'SubstitutionOn')&(df.teamId==teamid)]['name'].tolist()
	timeon = df[(df.type_displayName == 'SubstitutionOn')&(df.teamId==teamid)]['expandedMinute'].tolist()
	majoritylist = []
	minoritylist = []
	for i in range(len(timeon)):
		if((timeon[i]>=min1)&(timeon[i]<=max1)):
			player1min = timeon[i] - min1
			player2min = max1 - timeon[i]
			if(player1min >= player2min):
				majoritylist.append(playersubbedoff[i])
				minoritylist.append(playersubbedon[i])
			else:
				majoritylist.append(playersubbedon[i])
				minoritylist.append(playersubbedoff[i])
	players = list(set(allplayers) - set(minoritylist))
#     return players
	shirtNo = []
	for p in players:
		shirtNo.append(int(df[df.name==p]['shirtNo'].values[0]))

	passdf = df.query("(type_displayName=='Pass')&(name in @players)&(receiver in @players)&\
						 (outcomeType_displayName == 'Successful')&(teamId==@teamid)")
	passdf['x'] = passdf['x']
	passdf['y'] = 80 - passdf['y']
	passdf['endX'] = passdf['endX']
	passdf['endY'] = 80 - passdf['endY']
	cols = ['name','receiver']
	gamedf = passdf[cols]
	totalpassdf = gamedf.groupby(cols).size().reset_index(name="count")
	totalpassdf[cols] = np.sort(totalpassdf[cols],axis=1)
	totalpassdf = totalpassdf.groupby(cols)['count'].agg(['sum']).reset_index()
	avg_x = []
	avg_y = []
	blank=np.zeros(len(totalpassdf))
	totalpassdf['passer_x'] = blank
	totalpassdf['passer_y'] = blank
	totalpassdf['receiver_x'] = blank
	totalpassdf['receiver_y'] = blank
	uniquenames = np.array(players)
	uniquejerseys = np.array(shirtNo)
	totalpasses = []
	for name in uniquenames:
		player_pass_df = passdf.query("(name == @name)")
		x = np.mean(player_pass_df['x'])
		y = np.mean(player_pass_df['y'])
		totalpasses.append(len(player_pass_df))
		avg_x.append(x)
		avg_y.append(y)
	for i in range(len(totalpassdf)):
		passername = totalpassdf.iloc[i,0]
		receivername = totalpassdf.iloc[i,1]
		totalpassdf.iloc[i,3] = avg_x[np.where(uniquenames==passername)[0][0]]
		totalpassdf.iloc[i,4] = avg_y[np.where(uniquenames==passername)[0][0]]
		totalpassdf.iloc[i,5] = avg_x[np.where(uniquenames==receivername)[0][0]]
		totalpassdf.iloc[i,6] = avg_y[np.where(uniquenames==receivername)[0][0]]
		link = totalpassdf.iloc[i,2]
		passerx = totalpassdf.iloc[i,3]
		receiverx = totalpassdf.iloc[i,5]
		passery = totalpassdf.iloc[i,4]
		receivery = totalpassdf.iloc[i,6]
		if(link>=1):
			ax.plot([passerx, receiverx],[passery, receivery],linewidth=link/2.5, color ='lightgrey')
	for indx in range(len(uniquenames)):
		name = uniquenames[indx]
		size = 50*totalpasses[np.where(uniquenames==name)[0][0]]
		avgx = avg_x[np.where(uniquenames==name)[0][0]]
		avgy = avg_y[np.where(uniquenames==name)[0][0]]
		jersey = shirtNo[np.where(uniquenames==name)[0][0]]
		ax.scatter(avgx,avgy,color='#d7191c',s=1000,zorder=3)
		ax.annotate(jersey, (avgx, avgy),alpha=1,fontsize=25,color='w',
						horizontalalignment='center',
						verticalalignment='center').set_path_effects([path_effects.Stroke(linewidth=2,
																			foreground='black'), path_effects.Normal()])
	ax.text(125,3,'#    Player (Acc passes)',fontsize=25)
	for i in range(12):
		ax.text(127,3+4*i,'|',fontsize=50)
	for i in range(30):
		ax.text(125+i,3.5,'_',fontsize=50)
#     ax.plot([126,3],[126,50])
	for indx in range(len(players)):
		ax.text(125,7+4*indx,str(shirtNo[indx]),fontsize=25)
	for indx in range(len(players)):
		ax.text(130,7+4*indx,players[indx]+'('+str(totalpasses[indx])+')',fontsize=25)
	ax.set_title(teamname+' Passmap, Exp. Minutes '+str(min1)+ " to "+str(max1),fontsize=50)
	fig.text(0.02, 0.05, "Created by Soumyajit Bose / @Soumyaj15209314",fontstyle="italic",fontsize=20,color='#edece9')
	fig.text(0.02, -0.05, "Acc passes mentioned in brackets next to player names "+"\n"
			 "Width of line is proportional to number of passes exchanged between two players"+"\n"+
			 "A minimum of 1 pass need to be exchanged to show up as a line",
			 fontstyle="italic",fontsize=25,color='#edece9')
	add_image(bu_alt, fig, left=0.9, bottom=-0.05, width=0.07)
	return fig

st.title("Match Reports")
st.write('\n by Soumyajit Bose')
matchdf = pd.read_csv('Games.csv',index_col=[0]).reset_index(drop=True)
leagues = st.sidebar.selectbox('Choose League',matchdf.competition.unique().tolist())
matches = st.sidebar.selectbox('Choose Game',matchdf[matchdf.competition==leagues].gamename.tolist())
filename = matchdf.loc[(matchdf.competition==leagues)&(matchdf.gamename==matches),'filename'].values.item()
match, matchdict = filereader(filename)
hometeam = matchdict['home']['name']
awayteam = matchdict['away']['name']
homeid = matchdict['home']['teamId']
awayid = matchdict['away']['teamId']
teams = st.sidebar.selectbox('Choose team',[hometeam,awayteam,'Common'])
if(teams==hometeam):
	teamid = homeid
elif(teams==awayteam):
	teamid = awayid
else:
	teamid = 'None'
match = convert_to_actions(match)
match = match.query("(type_displayName not in ['Start','End','FormationChange','FormationSet'])").reset_index(drop=True)
match['name'] = match['name'].fillna('')
if((teams==hometeam)|(teams==awayteam)):
	viztype = st.sidebar.selectbox('Choose visualisation type',['Pass maps','Progressive Passes','Dangerous Passes', 
				'Defence Maps', 'Touch Heatmaps', 'Bad Passes'])
else:
	viztype = st.sidebar.selectbox('Choose visualisation type',['Data Table','Turnovers','Take Ons', 'Switches',  
				'Field Tilt', 'Key Passes'])
if(viztype=='Dangerous Passes'):
	fig = passtypes(match,teams,teamid)
	st.pyplot(fig)
elif(viztype=='Data Table'):
	fig = datatable(match,hometeam,homeid,awayteam,awayid)
	st.pyplot(fig)
elif(viztype=='Defence Maps'):
	fig = defensesmap(match,teams,teamid)
	st.pyplot(fig)
elif(viztype=='Bad Passes'):
	fig = badpasses(match,teams,teamid)
	st.pyplot(fig)
elif(viztype=='Turnovers'):
	fig = turnovermap(match,hometeam,homeid,awayteam,awayid)
	st.pyplot(fig)
elif(viztype=='Key Passes'):
	fig = keypasses(match,hometeam,homeid,awayteam,awayid)
	st.pyplot(fig)
elif(viztype=='Field Tilt'):
	fig = plotter(match,hometeam,homeid,awayteam,awayid)
	st.pyplot(fig)
elif(viztype=='Progressive Passes'):
	fig = progpassplotter(match,teams,teamid)
	st.pyplot(fig)
elif(viztype=='Switches'):
	fig = switches(match,hometeam,homeid,awayteam,awayid)
	st.pyplot(fig)
elif(viztype=='Take Ons'):
	fig = takeonmap(match,hometeam,homeid,awayteam,awayid)
	st.pyplot(fig)
elif(viztype=='Touch Heatmaps'):
	fig = touchmap(match,teams,teamid)
	st.pyplot(fig)
elif(viztype=='Pass maps'):
	mintime = min(match.expandedMinute)
	maxtime = max(match.expandedMinute)
	x = st.slider("Expanded Minutes", mintime, maxtime, (mintime, maxtime), 0.5)
	fig = passmap(match,teams,teamid,x[0],x[1])
	st.pyplot(fig)