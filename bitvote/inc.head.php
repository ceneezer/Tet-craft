<?php 
SESSION_START();
require '.cfg.php';
require_once '/srv/www/inc.MY_LOG.php';
require_once '/srv/www/inc.SQLATE.php';
require '.cfg.php';
$LANG='en';
if (substr($_SERVER['HTTP_HOST'],0,3)=="fr.")
   $LANG="fr";
if (isset($_REQUEST['lang']) && $_REQUEST['lang']=='fr')
   $LANG="fr";
switch ($LANG) {
   case 'FR':
      $meta_desc='Preuve de concept de BitVote : vote haché sur la blockchain'; //<155 chrs
      break;
   default:
      $meta_desc='BitVote proof-of-concept: hashed blockchain voting'; //<155 chrs
}
if (isset($_REQUEST['fbclid']))
MY_LOG('FBClick: '.$_REQUEST['fbclid']);

$PAGE=(isset($_REQUEST['page']))?$_REQUEST['page']:'';
$ID=(isset($_REQUEST['id']))?$_REQUEST['id']:0;
switch($PAGE) {
   case '':
      $title=$SITE_NAME;
      break;
   case 'issue':
      $title=$SITE_NAME.' '.SQLATE('motion', $LANG, true);
      if ($ID>0)
         $title.=' #'.$ID;
      else
         $title=$SITE_NAME.' '.SQLATE('new', $LANG, true).' '.$title;
      break;
   case 'tos':
      $title=$SITE_NAME.' '.SQLATE('terms of service', $LANG, true);
      break;
   case 'privacy':
      $title=$SITE_NAME.' '.SQLATE('privacy policy', $LANG, true);
      break;
   case 'log';
      $title=$SITE_NAME.' '.SQLATE('blockchain', $LANG, true);
      break;
   case 'issues':
      $title=$SITE_NAME.' '.SQLATE('recent motions', $LANG, true);
      break;
   case 'Profile':
      $title=$SITE_NAME.' '.SQLATE('profile', $LANG, true);
      if ($ID>0)
         $title.=' #'.$ID;
      break;
      case 'mine':
      case 'me':
         $title=$SITE_NAME.' '.SQLATE('rankings', $LANG, true);
      break;
}?><!DOCTYPE html><html lang="<?=$LANG?>"><head>
   <meta charset="utf-8" />
   <meta name="viewport" content="width=device-width, initial-scale=1.0" />
   <meta http-equiv="X-UA-Compatible" content="ie=edge" />
   <link rel="stylesheet" href="pub/style.css" /> 
<!--   <script src="pub/sorttable.js"></script> -->
<?php require_once '/srv/www/inc.meta.php'; ?>
</head><body>
   <img id="bg" src="pub/pics/Cansenate.jpg" />
   <header>
      <h1><?=$SITE_NAME?><img id="logo" src="favicon.ico" /></h1>
      <form class="search" method="post">
      <input type="submit" value="<?=SQLATE('search', $LANG, true)?>" />
<input name="search" value="<?=isset($_REQUEST['search'])?$_REQUEST['search']:''?>" placeholder="<?=SQLATE('search', $LANG, true)?>" />
      </form>
      <aside class="flags en">
<a href="https://<?=$site?>/?page=<?=$PAGE?>&id=<?=$ID?>" title="English"><img src="pub/pics/en.jpg"/></a>
      </aside>
      <aside class="flags fr">
<a href="https://fr.<?=$site?>/?page=<?=$PAGE?>&id=<?=$ID?>" title="French"><img src="pub/pics/fr.svg"/></a>
      </aside>
      <nav>
         <ul>
            <li><a href="?page=issues"><?=SQLATE('motions', $LANG, true)?></a></li>
            <li><a href="?page=log"><?=SQLATE('blockchain', $LANG, true)?></a></li>
<?php if (isset($_REQUEST['UID'])) { ?>
            <li><a href="?page=me"><?=SQLATE('rankings', $LANG, true)?></a></li>
<?php } ?>
         </ul>
      </nav>
   </header>