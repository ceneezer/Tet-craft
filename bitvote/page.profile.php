<?php
   $sql='SELECT u.id, u.ssid, u.fname, u.lname, u.email, u.phone, p.';
   if ($LANG=='fr')
      $sql.='fr';
   $sql.='caption as party, u.link FROM users AS u LEFT JOIN parties AS p ON p.id=u.ID_party WHERE u.link IS NOT NULL AND u.id="'.SQLIFY($ID).'"';
   $sql=SQL2ARRAY($sql);
   if (count($sql)>0) {
      $sql=$sql[0];
?>
   <main>
      <h3>
<?php if ($sql['ssid']>0) { ?>
<a href="https://sencanada.ca/<?=($LANG=='fr')?'fr/dans-la-chambre':'en/in-the-chamber'?>/votes/senator/<?=$sql['ssid']?>/" target="_blank">
<?php } else { ?>
<a href='<?=$sql['link']?>' target="_blank">
<?php } ?>
<?=$sql['fname']?> <?=$sql['lname']?></a> 
<?=$sql['phone']?> <?=$sql['email']?> <?=$sql['party']?>
      </h3>
<?php 
   $sql='SELECT COUNT(vf.Vvalue)-COUNT(va.Vvalue) as score FROM issue_votes AS i LEFT JOIN issue_votes AS b ON b.ID_issue=i.ID_issue AND b.ID_user="'.SQLIFY($ID).'" LEFT JOIN votes AS vf ON vf.id=i.ID_vote AND i.ID_vote=b.ID_vote LEFT JOIN votes AS va ON va.id=i.ID_vote AND NOT i.ID_vote=b.ID_vote WHERE b.ID_user="'.SQLIFY($ID).'"';
   if (isset($_SESSION['UID']))
      $sql.=' AND i.ID_user="'.SQLIFY($_SESSION['UID']).'"';
   $sql=SQL2ARRAY($sql);
   if (count($sql)>0) {
      $sql=$sql[0];
?>
      <h4>
<?=(isset($_SESSION['UID']))?SQLATE('personal', $LANG, true).' ':''?><?=SQLATE('ranking', $LANG, true)?>: <?=$sql['score']?>
      </h4>
<?php } ?>
      <section>
         <table id="tblVotes">
            <tr>
               <th onmouseup="this.classList.remove('wait');" onmousedown="this.classList.add('wait');" onclick="sortTable('tblVotes', 0)"><?=SQLATE('motion', $LANG, true)?></th>
               <th onmouseup="this.classList.remove('wait');" onmousedown="this.classList.add('wait');" onclick="sortTable('tblVotes', 1)"><?=SQLATE('vote', $LANG, true)?></th>
               <th onmouseup="this.classList.remove('wait');" onmousedown="this.classList.add('wait');" onclick="sortTable('tblVotes', 2)"><?=SQLATE('date', $LANG, true)?></th>
            </tr>
<?php
   $sql='SELECT i.';
   if ($LANG=='fr')
      $sql.='fr';
   $sql.='title as title, iv.ID_issue, v.';
   if ($LANG=='fr')
      $sql.='fr';
   $sql.='caption as caption, DATE_FORMAT(i.deadline, "%Y-%m-%e") AS d FROM issue_votes AS iv LEFT JOIN votes AS v ON v.id=iv.ID_vote LEFT JOIN issues AS i ON i.id=iv.ID_issue WHERE iv.ID_user="'.SQLIFY($ID).'"';
   if (isset($_REQUEST['search']) && $_REQUEST['search']!='') {
      $search=str_replace('%', '', $search);
      $search=SQLIFY($_REQUEST['search']);
      $sql.=' AND (';
      $sql.='i.title LIKE "%'.$search.'%"';
      $sql.=' OR i.frtitle LIKE "%'.$search.'%"';
      $sql.=' OR i.journal LIKE "%'.$search.'%"';
      $sql.=' OR i.leginfo LIKE "%'.$search.'%"';
      $sql.=' OR i.ssid LIKE "%'.$search.'%"';
      $sql.=' OR i.mpid LIKE "%'.$search.'%"';
      $sql.=' OR i.id LIKE "%'.$search.'%"';
      $sql.=' OR v.caption LIKE "%'.$search.'%"';
      $sql.=' OR v.frcaption LIKE "%'.$search.'%"';
      $sql.=')';
   }
   $sql.=' ORDER BY i.deadline DESC';
   $sql=SQL2ARRAY($sql);
   foreach ($sql as $s) {
?>
            <tr>
               <td style="text-align:left;" ><a href="?page=issue&id=<?=$s['ID_issue']?>"><?=$s['title']?></a></td>
               <td><?=$s['caption']?></td>
               <td style="width:10em"><span><?=$s['d']?></span></td>
            </tr>
<?php } ?>
         </table>
      </section>
   </main>
<?php } ?>